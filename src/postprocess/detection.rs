// 该文件是 Shanan CV 项目的一部分。
// src/postprocess/detection.rs - 针对目标检测算法的后处理相关功能
//
// 本文件根据 Apache 许可证第 2.0 版（以下简称“许可证”）授权使用；
// 除非遵守该许可证条款，否则您不得使用本文件。
// 您可通过以下网址获取许可证副本：
// http://www.apache.org/licenses/LICENSE-2.0
// 除非适用法律要求或书面同意，根据本许可协议分发的软件均按“原样”提供，
// 不附带任何形式的明示或暗示的保证或条件。
// 有关许可权限与限制的具体条款，请参阅本许可协议。
//
// Copyright (C) 2026 Johann Li <me@qinka.pro>, Wareless Group

use cubecl::{CubeScalar, num_traits::Zero, prelude::*};
use thiserror::Error;

use crate::{data::DataBuffer, kernel::sigmoid};

#[derive(Debug, Error)]
pub enum Yolo26Error {
  #[error("无效的输入形状: {0}")]
  InvalidInputShape(String),
  #[error("运行时错误: {0}")]
  LaunchError(#[from] LaunchError),
}
pub struct Yolo26Config {
  width: u32,
  height: u32,
  dim: u32,
}

impl Default for Yolo26Config {
  fn default() -> Self {
    Self {
      width: 640,
      height: 640,
      dim: 1,
    }
  }
}

impl Yolo26Config {
  pub fn with_shape(mut self, width: u32, height: u32) -> Self {
    self.width = width;
    self.height = height;
    self
  }

  pub fn build(self) -> Result<Yolo26, Yolo26Error> {
    Ok(Yolo26 {
      width: self.width,
      height: self.height,
      dim: self.dim,
    })
  }

  pub fn with_dim(mut self, dim: u32) -> Self {
    self.dim = dim;
    self
  }
}

pub struct Yolo26 {
  width: u32,
  height: u32,
  dim: u32,
}

pub type PPResult<R, F, I> = (DataBuffer<R, F>, DataBuffer<R, I>, DataBuffer<R, F>);

impl Yolo26 {
  /// 执行后处理操作
  /// cls: 分类结果，形状为 [N, num_classes, H, W]
  /// reg: 回归结果，形状为 [N, 4, H, W]
  /// 返回 (score, index, bbox) 三个张量，分别是分类得分、类别索引和边界框坐标
  pub fn execute<R: Runtime, F: Float + CubeElement, I: Int + CubeElement>(
    &self,
    client: &ComputeClient<R>,
    cls: DataBuffer<R, F>,
    reg: DataBuffer<R, F>,
    stride: F,
  ) -> Result<PPResult<R, F, I>, Yolo26Error> {
    let [n, c, h, w] = *cls.shape() else {
      return Err(Yolo26Error::InvalidInputShape(
        "分类结果张量形状不正确，预期为 [N, num_classes, H, W]".to_string(),
      ));
    };

    let cls_sigmoid = cls.empty_like(client);

    let count = n * c * h * w / self.dim as usize;
    sigmoid::launch::<F, R>(
      client,
      CubeCount::Static(count as u32, 1, 1),
      CubeDim::new_1d(self.dim),
      cls.into_tensor_arg(1),
      cls_sigmoid.into_tensor_arg(1),
    )?;

    let score: DataBuffer<R, F> = DataBuffer::with_shape(&[n, h, w], client);
    let index: DataBuffer<R, I> = DataBuffer::with_shape(&[n, h, w], client);

    let count = n * h * w / self.dim as usize;
    classify::launch::<F, I, R>(
      client,
      CubeCount::Static(count as u32, 1, 1),
      CubeDim::new_1d(self.dim),
      cls_sigmoid.into_tensor_arg(1),
      score.into_tensor_arg(1),
      index.into_tensor_arg(1),
    )?;

    let bbox: DataBuffer<R, F> = DataBuffer::with_shape(&[n, 4, h, w], client);

    bbox::launch::<F, R>(
      client,
      CubeCount::Static(count as u32, 1, 1),
      CubeDim::new_1d(self.dim),
      reg.into_tensor_arg(1),
      bbox.into_tensor_arg(1),
      ScalarArg::new(F::new(self.width as f32)),
      ScalarArg::new(F::new(self.height as f32)),
      ScalarArg::new(stride),
    )?;

    Ok((score, index, bbox))
  }
}

/// 将 Yolo 检测结果中的分类指标进行处理，输出每个位置的最大分类得分和对应的类别索引
///
/// cls: 输入分类结果，形状为 [N, num_classes, H, W], 应该已经调用过 sigmoid 激活函数
/// score: 输出分类结果得分 [N, H, W]
/// index: 输出分类结果类型索引 [N, H, W]
#[cube(launch)]
fn classify<F: Float, I: Int>(cls: Tensor<F>, score: &mut Tensor<F>, index: &mut Tensor<I>) {
  // 输出张量总元素 = N * H * W
  let nhw = score.len();

  // 线程全局索引
  let idx = ABSOLUTE_POS;
  if idx < nhw {
    // 获取输入维度
    let c_dim = cls.shape(1);
    let h_dim = cls.shape(2);
    let w_dim = cls.shape(3);

    // 将 idx 映射回 (n, h, w)
    // idx = n * H * W + h * W + w
    let hw = h_dim * w_dim;
    let n_idx = idx / hw;
    let rem = idx % hw;
    let h_idx = rem / w_dim;
    let w_idx = rem % w_dim;

    // 输入 strides (支持任意 stride 布局)
    let stride_n = cls.stride(0);
    let stride_c = cls.stride(1);
    let stride_h = cls.stride(2);
    let stride_w = cls.stride(3);

    // 计算 base offset (c=0 时的位置)
    let base = n_idx * stride_n + h_idx * stride_h + w_idx * stride_w;

    // 初始化: c=0 的值
    let mut best_c = 0;
    let mut best_val = cls[base];

    for c in 1..c_dim {
      let off = base + c * stride_c;
      let v = cls[off];
      if v > best_val {
        best_val = v;
        best_c = c;
      }
    }

    // 写入输出: 最大值 + 对应通道索引
    score[idx] = best_val;
    index[idx] = I::cast_from(best_c);
  }
}

/// 将 Yolo 检测结果中的回归指标进行处理，输出每个位置的边界框坐标
/// reg: 输入回归结果，形状为 [N, 4, H, W], 包含 (cx, cy, w, h) 四个通道
/// bbox: 输出边界框坐标，形状为 [N, 4, H, W] 为 xmin, ymin, xmax, ymax
#[cube(launch)]
fn bbox<F: Float + CubeScalar + Zero>(
  reg: Tensor<F>,
  bbox: &mut Tensor<F>,
  image_width: F,
  image_height: F,
  stride: F,
) {
  // 输出张量总元素 = N * H * W
  let nhw = bbox.len() / 4; // 每个位置有4个坐标

  // 线程全局索引
  let idx = ABSOLUTE_POS;
  if idx < nhw {
    let half_value = F::new(comptime!(0.5));
    let zero_value = F::new(comptime!(0.0));

    // 获取输入维度
    let h_dim = reg.shape(2);
    let w_dim = reg.shape(3);

    // 将 idx 映射回 (n, h, w)
    // idx = n * H * W + h * W + w
    let hw = h_dim * w_dim;
    let n_idx = idx / hw;
    let rem = idx % hw;
    let h_idx = rem / w_dim;
    let w_idx = rem % w_dim;

    // 输入 strides (支持任意 stride 布局)
    let stride_n = reg.stride(0);
    let stride_c = reg.stride(1);
    let stride_h = reg.stride(2);
    let stride_w = reg.stride(3);

    // 计算 base offset (c=0 时的位置)
    let base = n_idx * stride_n + h_idx * stride_h + w_idx * stride_w;

    // 获取回归值
    let cx = reg[base]; // c=0
    let cy = reg[base + stride_c]; // c=1
    let cw = reg[base + 2 * stride_c]; // c=2
    let ch = reg[base + 3 * stride_c]; // c=3

    let grid_x = F::cast_from(w_idx) + half_value;
    let grid_y = F::cast_from(h_idx) + half_value;

    let xmin = cubecl::prelude::clamp((grid_x - cx) * stride, zero_value, image_width);
    let ymin = cubecl::prelude::clamp((grid_y - cy) * stride, zero_value, image_height);
    let xmax = cubecl::prelude::clamp((grid_x + cw) * stride, zero_value, image_width);
    let ymax = cubecl::prelude::clamp((grid_y + ch) * stride, zero_value, image_height);

    // 转换为边界框坐标 (xmin, ymin, xmax, ymax)
    bbox[idx] = (xmin / image_width).clamp(F::new(0.0), F::new(1.0)); // xmin
    bbox[idx + nhw] = (ymin / image_height).clamp(F::new(0.0), F::new(1.0)); // ymin
    bbox[idx + 2 * nhw] = (xmax / image_width).clamp(F::new(0.0), F::new(1.0)); // xmax
    bbox[idx + 3 * nhw] = (ymax / image_height).clamp(F::new(0.0), F::new(1.0)); // ymax
  }
}
