use std::io::{self, Write};

use opencv::prelude::*;
use opencv::highgui;
use opencv::types;
use opencv::core;
use opencv::Result;
use opencv::imgproc;
use opencv::core::Mat;
use opencv::videoio::VideoCapture;
use opencv::objdetect;

mod utils;
use crate::utils::write_image;

fn main() -> opencv::Result<()> {
    // Initialize OpenCV
    highgui::named_window("Webcam", highgui::WINDOW_NORMAL)?;

    let mut webcam = VideoCapture::new(1, 0)?;

    // Load the cascade classifier XML contents
    let frontal_face_content = include_str!("../haarcascades/haarcascade_frontalface_alt.xml");
    let eye_content = include_str!("../haarcascades/haarcascade_eye.xml");
    let full_body_content = include_str!("../haarcascades/haarcascade_fullbody.xml");
    let cat_face_content = include_str!("../haarcascades/haarcascade_frontalcatface_extended.xml");

    // Create temporary files and write XML contents to them
    let frontal_face_temp_path = utils::write_temp_file("haarcascade_frontalface_alt.xml", frontal_face_content)?;
    let eye_temp_path = utils::write_temp_file("haarcascade_eye.xml", eye_content)?;
    let full_body_temp_path = utils::write_temp_file("haarcascade_fullbody.xml", full_body_content)?;
    let cat_face_temp_path = utils::write_temp_file("haarcascade_frontalcatface_extended.xml", cat_face_content)?;

    // Load the cascade classifiers using the temporary file paths
    let mut cascade_frontal_face = objdetect::CascadeClassifier::new(&frontal_face_temp_path)?;
    let mut cascade_eye = objdetect::CascadeClassifier::new(&eye_temp_path)?;
    let mut cascade_full_body = objdetect::CascadeClassifier::new(&full_body_temp_path)?;
    let mut cascade_cat_face = objdetect::CascadeClassifier::new(&cat_face_temp_path)?;


    loop {
        let mut frame = Mat::default();
        webcam.read(&mut frame)?;

        if frame.size()?.width > 0 {
            let mut frame_clone = frame.clone();

            detect_faces(&mut frame_clone, &mut cascade_frontal_face)?;
            detect_eyes(&mut frame_clone, &mut cascade_eye)?;
            detect_bodies(&mut frame_clone, &mut cascade_full_body)?;
            detect_cat_face(&mut frame_clone, &mut cascade_cat_face)?;

            highgui::imshow("Webcam", &frame_clone)?;

            highgui::wait_key(1)?;
        }
    }

    Ok(())
}

fn detect_bodies(frame: &mut Mat, cascade: &mut objdetect::CascadeClassifier) -> Result<()> {
  let min_size: core::Size = core::Size::new(50, 100);
  let max_size: core::Size = core::Size::new(0, 0);
  let scale = 1.1;
  let min_neighbors = 5;
  let flags = 1;

  // Convert the frame to grayscale for detection
  let mut gray = Mat::default();
  imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

  // Detect full bodies in the frame
  let mut bodies = types::VectorOfRect::new();
  cascade.detect_multi_scale(&gray, &mut bodies, scale, min_neighbors, flags, min_size, max_size)?;

  // Draw rectangles around detected bodies
  for body in bodies.iter() {
    imgproc::rectangle(frame, body, core::Scalar::new(0.0, 0.0, 255.0, 0.0), 2, 8, 0)?;
  }

  // Print a line if bodies are detected
  if !bodies.is_empty() {
    println!("Body detected!");
    write_image("body", frame)?;
  }

  Ok(())
}

fn detect_eyes(frame: &mut Mat, cascade: &mut objdetect::CascadeClassifier) -> Result<()> {
  let min_size: core::Size = core::Size::new(5, 5);
  let max_size: core::Size = core::Size::new(0, 0);
  let scale = 1.1;
  let min_neighbors = 70;
  let flags = 1;

    // Convert the frame to grayscale for detection
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Detect full bodies in the frame
    let mut bodies = types::VectorOfRect::new();
    cascade.detect_multi_scale(&gray, &mut bodies, scale, min_neighbors, flags, min_size, max_size)?;

    // Draw rectangles around detected bodies
    for body in bodies.iter() {
      imgproc::rectangle(frame, body, core::Scalar::new(255.0, 0.0, 0.0, 0.0), 2, 8, 0)?;
    }

    // Print a line if bodies are detected
    if !bodies.is_empty() {
      println!("Eye detected!");
      write_image("eye", frame)?;
    }
    Ok(())
}

fn detect_faces(frame: &mut Mat, cascade: &mut objdetect::CascadeClassifier) -> Result<()> {
  let min_size: core::Size = core::Size::new(0, 0);
  let max_size: core::Size = core::Size::new(0, 0);
  let scale = 1.1;
  let min_neighbors = 5;
  let flags = 1;

  // Convert the frame to grayscale for detection
  let mut gray = Mat::default();
  imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

  // Detect full bodies in the frame
  let mut bodies = types::VectorOfRect::new();
  cascade.detect_multi_scale(&gray, &mut bodies, scale, min_neighbors, flags, min_size, max_size)?;

  // Draw rectangles around detected bodies
  for body in bodies.iter() {
    imgproc::rectangle(frame, body, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, 8, 0)?;
  }

  // Print a line if bodies are detected
  if !bodies.is_empty() {
    println!("Face detected!");
    utils::write_image("face", frame)?;
  }

  Ok(())
}

fn detect_cat_face(frame: &mut Mat, cascade: &mut objdetect::CascadeClassifier) -> Result<()> {
  let min_size: core::Size = core::Size::new(0, 0);
  let max_size: core::Size = core::Size::new(0, 0);
  let scale = 1.1;
  let min_neighbors = 5;
  let flags = 1;

    // Convert the frame to grayscale for detection
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Detect full bodies in the frame
    let mut bodies = types::VectorOfRect::new();
    cascade.detect_multi_scale(&gray, &mut bodies, scale, min_neighbors, flags, min_size, max_size)?;

    // Draw rectangles around detected bodies
    for body in bodies.iter() {
      imgproc::rectangle(frame, body, core::Scalar::new(255.0, 0.0, 0.0, 0.0), 2, 8, 0)?;
    }

    // Print a line if bodies are detected
    if !bodies.is_empty() {
      println!("Cat detected!");
      write_image("cat", frame)?;
    }
    Ok(())
}

