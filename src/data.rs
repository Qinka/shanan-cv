// 该文件是 Shanan CV 项目的一部分。
// src/data.rs - 对数据张量的表示形式
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

use cubecl::{prelude::*, server::Handle, std::tensor::compact_strides};
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Debug)]
pub struct DataBuffer<R: Runtime, T: CubeElement> {
  data: Handle,
  shape: Vec<usize>,
  strides: Vec<usize>,
  _r: PhantomData<R>,
  _t: PhantomData<T>,
}

#[derive(Debug, Error)]
pub enum DataBufferError {
  #[error("数据创建错误: {0}")]
  CreationError(String),
  #[error("无效的形状: {0}")]
  InvalidShape(String),
  #[error("无效的数据格式: {0}")]
  InvalidData(String),
  #[error("运行时错误: {0}")]
  RuntimeError(String),
}

impl<R: Runtime, T: CubeElement> Clone for DataBuffer<R, T> {
  fn clone(&self) -> Self {
    Self {
      // Handle is a pointer to the data, so cloning it is cheap
      data: self.data.clone(),
      shape: self.shape.clone(),
      strides: self.strides.clone(),
      _r: PhantomData,
      _t: PhantomData,
    }
  }
}

impl<R: Runtime, T: CubeElement + CubePrimitive> DataBuffer<R, T> {
  pub fn from_slice(
    data: &[T],
    shape: &[usize],
    client: &ComputeClient<R>,
  ) -> Result<Self, DataBufferError> {
    let strides = compact_strides(shape);
    let handle = client.create_from_slice(T::as_bytes(data));
    Ok(Self {
      data: handle,
      shape: shape.to_vec(),
      strides,
      _r: PhantomData,
      _t: PhantomData,
    })
  }

  pub fn shape(&self) -> &[usize] {
    &self.shape
  }

  pub fn strides(&self) -> &[usize] {
    &self.strides
  }

  /// 形成类型的尺寸和数据格式的对象
  pub fn empty_like(&self, client: &ComputeClient<R>) -> Self {
    let handle = client.empty(self.data.size() as usize);
    Self {
      data: handle,
      shape: self.shape.clone(),
      strides: self.strides.clone(),
      _r: PhantomData,
      _t: PhantomData,
    }
  }

  /// 根据指定的形状和数据格式创建一个新的 DataBuffer
  pub fn with_shape(shape: &[usize], client: &ComputeClient<R>) -> Self {
    let strides = compact_strides(shape);
    let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
    let handle = client.empty(size);
    Self {
      data: handle,
      shape: shape.to_vec(),
      strides,
      _r: PhantomData,
      _t: PhantomData,
    }
  }

  pub fn into_tensor_arg(&self, line_size: usize) -> TensorArg<'_, R> {
    unsafe { TensorArg::from_raw_parts::<T>(&self.data, &self.strides, &self.shape, line_size) }
  }

  pub fn into_vec(self, client: &ComputeClient<R>) -> Result<Vec<T>, DataBufferError> {
    let bytes = client.read_one(self.data);
    Ok(T::from_bytes(&bytes).to_vec())
  }
}

pub trait ToDataBuffer<R: Runtime, T: CubeElement> {
  fn to_data_buffer(&self) -> DataBuffer<R, T>;
}

pub trait FromDataBuffer<R: Runtime, T: CubeElement>: Sized {
  fn from_data_buffer(buffer: DataBuffer<R, T>) -> Self;
}
