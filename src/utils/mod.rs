use std::io::{self, Write};
use tempfile::{Builder, NamedTempFile};
use std::env;
use std::fs;
use std::fs::{File, Permissions};
use std::os::unix::fs::PermissionsExt;

use chrono::Local;
use dirs::home_dir;
use opencv::core;
use opencv::core::Mat;
use opencv::{Error, Result, imgcodecs};

#[derive(Debug)]
pub enum CustomError {
    IoError(std::io::Error),
    OpenCvError(opencv::Error),
}

impl From<std::io::Error> for CustomError {
    fn from(io_err: std::io::Error) -> Self {
        CustomError::IoError(io_err)
    }
}

impl From<opencv::Error> for CustomError {
    fn from(cv_err: opencv::Error) -> Self {
        CustomError::OpenCvError(cv_err)
    }
}

impl std::error::Error for CustomError {}

impl std::fmt::Display for CustomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CustomError::IoError(inner) => inner.fmt(f),
            CustomError::OpenCvError(inner) => inner.fmt(f),
        }
    }
}

impl From<CustomError> for opencv::Error {
    fn from(custom_err: CustomError) -> Self {
        opencv::Error::new(opencv::core::StsError as i32, format!("Custom error: {}", custom_err))
    }
}

pub fn write_temp_file(filename: &str, content: &str) -> Result<String, CustomError> {
    let temp_dir = env::temp_dir();

    let mut temp_file = Builder::new()
      .prefix(filename)
      .tempfile_in(temp_dir)?;

    let temp_file_path = temp_file.path().to_string_lossy().into_owned();

    temp_file.write_all(content.as_bytes())?;
    temp_file.flush()?;

    // Set the desired permissions on the temporary file
    let mut permissions = std::fs::metadata(&temp_file_path)?.permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(&temp_file_path, permissions)?;

    // Copy the temporary file to the desired location
    let destination_path = format!("{}/{}", copy_directory().to_owned(), filename);
    delete_file_if_exists(&destination_path)?;
    std::fs::copy(&temp_file_path, &destination_path)?;

    Ok(destination_path)
}

pub fn write_image(name: &str, frame: &mut Mat) -> Result<bool> {
  let now = Local::now();
  let destination_directory = format!("{}/Desktop/detections", home_dir().unwrap().display());

  if !directory_exists(&destination_directory) {
    create_directory(&destination_directory);
  }

  let image_name = format!("{}/{}_{}.jpg", destination_directory, name, now.format("%Y%m%d_%H%M%S"));
  imgcodecs::imwrite(&image_name, frame, &core::Vector::default())
}

fn copy_directory() -> String {
  let destination_directory = format!("{}/Desktop/camera", home_dir().unwrap().display());

  if !directory_exists(&destination_directory) {
    create_directory(&destination_directory);
  }

  destination_directory
}

fn directory_exists(path: &str) -> bool {
  std::fs::metadata(path)
    .map(|metadata| metadata.is_dir())
    .unwrap_or(false)
}

fn create_directory(path: &str) -> bool {
  match std::fs::create_dir_all(path) {
    Ok(_) => true,
    Err(_) => false,
  }
}

fn delete_file_if_exists(file_path: &str) -> Result<(), std::io::Error> {
    if fs::metadata(file_path).is_ok() {
        fs::remove_file(file_path)?;
    }

    Ok(())
}
