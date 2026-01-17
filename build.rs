//! this build script compiles GEM kernels
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("Building cuda source files for GEM...");
    println!("cargo:rerun-if-changed=csrc");

    #[cfg(feature = "cuda")] {
        let csrc_headers = ucc::import_csrc();
        let mut cl_cuda = ucc::cl_cuda();
        cl_cuda.ccbin(false);
        cl_cuda.flag("-lineinfo");
        cl_cuda.flag("-maxrregcount=128");

        // Optimize based on profile or env var
        let profile = std::env::var("PROFILE").unwrap_or("release".to_string());
        let opt_level_env = std::env::var("OPT_LEVEL").ok();
        
        let (debug_info, opt_lvl) = if let Some(lvl) = opt_level_env {
             (false, lvl.parse().unwrap_or(3)) 
        } else if profile == "release" {
            (false, 3)
        } else {
             // fast-compile or debug
            (true, 1) 
        };

        if profile == "fast-compile" {
             println!("cargo:warning=Building with fast-compile profile: CUDA opt_level={}", opt_lvl);
        }

        cl_cuda.debug(debug_info).opt_level(opt_lvl)
            .include(&csrc_headers)
            .files(["csrc/kernel_v1.cu"]);
        cl_cuda.compile("gemcu");
        println!("cargo:rustc-link-lib=static=gemcu");
        println!("cargo:rustc-link-lib=dylib=cudart");
        ucc::bindgen(["csrc/kernel_v1.cu"], "kernel_v1.rs");
        ucc::export_csrc();
        ucc::make_compile_commands(&[&cl_cuda]);
    }
}
