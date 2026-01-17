// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! This is an experimental interactive cutting-then-mapping
//! implementation.
//!
//! The key idea is to only repartition the endpoint groups that
//! are unable to be mapped.

use std::path::PathBuf;
use gem::repcut::RCHyperGraph;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::AIG;
use gem::staging::build_staged_aigs;
use gem::pe::{process_partitions, Partition};
use netlistdb::NetlistDB;
use netlistdb::GeneralHierName;
use rayon::prelude::*;

/// Call builtin partitioner.
fn run_par(hg: &RCHyperGraph, num_parts: usize, fast: bool) -> Vec<Vec<usize>> {
    clilog::debug!("invoking partitioner (#parts {})", num_parts);
	// Handle the special case where num_parts = 1
    // mt-kahypar requires k >= 2, so we handle k=1 manually
    if num_parts == 1 {
        let mut parts = vec![vec![]; 1];
        // Put all vertices in the single partition
        for i in 0..hg.num_vertices() {
            parts[0].push(i);
        }
        return parts;
    }
	
    let parts_ids = hg.partition(num_parts, fast);
    let mut parts = vec![vec![]; num_parts];
    for (i, part_id) in parts_ids.into_iter().enumerate() {
        parts[part_id].push(i);
    }
    parts
}

#[derive(clap::Parser, Debug)]
struct SimulatorArgs {
    /// Gate-level verilog path synthesized in our provided library.
    ///
    /// If your design is still at RTL level, you should synthesize it
    /// in yosys first.
    netlist_verilog: PathBuf,
    /// Top module type in netlist to analyze.
    ///
    /// If not specified, we will guess it from the hierarchy.
    #[clap(long)]
    top_module: Option<String>,
    /// Level split thresholds.
    #[clap(long, value_delimiter=',')]
    level_split: Vec<usize>,
    /// Output path for the serialized partitions.
    parts_out: PathBuf,
    /// The maximum allowance of layers for merging-induced degradations.
    ///
    /// By default is 0, meaning no degradation is allowed.
    #[clap(long, default_value_t=0)]
    max_stage_degrad: usize,
    /// The number of boomerang stages.
    #[clap(long, default_value_t=13)]
    stages: usize,
    /// Use fast (high-speed) partitioning instead of search-heavy quality/deterministic mode.
    #[clap(long, default_value_t=false)]
    fast_part: bool,
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::enable_timer("");
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let args = <SimulatorArgs as clap::Parser>::parse();
    clilog::info!("Simulator args:\n{:#?}", args);

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins()
    ).expect("cannot build netlist");

    let aig = AIG::from_netlistdb(&netlistdb);
    clilog::info!("netlist has {} pins, {} aig pins, {} and gates",
             netlistdb.num_pins, aig.num_aigpins, aig.and_gate_cache.len());

    let stageds = build_staged_aigs(&aig, &args.level_split);

    let stages_effective_parts = stageds.iter().map(|&(l, r, ref staged)| {
        clilog::info!("interactive partitioning stage {}-{}", l, match r {
            usize::MAX => "max".to_string(),
            r @ _ => format!("{}", r)
        });

        let mut parts_indices_good = Vec::new();
        // always made sure that staged output pins are at fronts.
        let mut unrealized_endpoints = (0..staged.num_endpoint_groups()).collect::<Vec<_>>();
        let mut division = 600;

        let timer_interactive_stage = clilog::stimer!("interactive_stage_loop");
        while !unrealized_endpoints.is_empty() {

            division = (division / 2).max(1);
            let num_parts = (unrealized_endpoints.len() + division - 1) / division;
            clilog::info!("current: {} endpoints, try {} parts", unrealized_endpoints.len(), num_parts);
            let staged_ur = staged.to_endpoint_subset(&unrealized_endpoints);
            let hg_ur = RCHyperGraph::from_staged_aig(&aig, &staged_ur);
            let timer_hg_part = clilog::stimer!("hg_partitioning");
            let mut parts_indices = run_par(&hg_ur, num_parts, args.fast_part);
            clilog::finish!(timer_hg_part);
            for idcs in &mut parts_indices {
                for i in idcs {
                    *i = unrealized_endpoints[*i];
                }
            }
            let timer_part_build = clilog::stimer!("partition_build_ones");
            let parts_try = parts_indices.par_iter()
                .map(|endpts| Partition::build_one(&aig, staged, endpts, args.stages))
                .collect::<Vec<_>>();
            clilog::finish!(timer_part_build);
            let mut new_unrealized_endpoints = Vec::new();
            for (idx, part_opt) in parts_indices.into_iter().zip(parts_try.into_iter()) {
                match part_opt {
                    Some(_part) => {
                        parts_indices_good.push(idx);
                    }
                    None => {
                        if idx.len() == 1 {
                            let endpt_id = idx[0];
                            // Map endpoint (EndpointGroup index) to AIG pin
                            // stagedAIG.get_endpoint_group(aig, endpt_id)
                            let edg = staged.get_endpoint_group(&aig, endpt_id);
                            // We can iterate inputs of the endpoint group to find a pin.
                            // Assuming typical endpoint is one pin.
                            // Format name based on EndpointGroup type
                            let edg_name = match edg {
                                gem::aig::EndpointGroup::PrimaryOutput(pin) => {
                                    let logic_pin = aig.pin_map[pin >> 1];
                                    if logic_pin != usize::MAX {
                                        let (h, n, i) = netlistdb.pin_name(logic_pin);
                                        format!("PO({}/{}{:?})", h, n, i)
                                    } else {
                                        format!("PO(AIG:{})", pin >> 1)
                                    }
                                },
                                gem::aig::EndpointGroup::DFF(dff) => {
                                    let aig_pin_idx = dff.q;
                                    let logic_pin = aig.pin_map[aig_pin_idx];
                                    if logic_pin != usize::MAX {
                                        let (h, n, i) = netlistdb.pin_name(logic_pin);
                                        format!("DFF({}/{}{:?})", h, n, i)
                                    } else {
                                        // Try to find cell name from driver
                                        if let gem::aig::DriverType::DFF(cellid) = aig.drivers[aig_pin_idx] {
                                            format!("DFF(Cell:{})", netlistdb.cellnames[cellid].dbg_fmt_hier())
                                        } else {
                                            format!("DFF(AIG:{})", aig_pin_idx)
                                        }
                                    }
                                },
                                gem::aig::EndpointGroup::RAMBlock(_) => "RAM".to_string(),
                                gem::aig::EndpointGroup::StagedIOPin(pin) => format!("StagedIO({})", pin),
                            };
                            
                            panic!("A single endpoint still cannot map: endpoint_id={} name={}. You need to increase level cut granularity.", endpt_id, edg_name);
                        }
                        for endpt_i in idx {
                            new_unrealized_endpoints.push(endpt_i);
                        }
                    }
                }
            }
            new_unrealized_endpoints.sort_unstable();
            unrealized_endpoints = new_unrealized_endpoints;
        }

        clilog::finish!(timer_interactive_stage);
        clilog::info!("interactive partition completed: {} in total. merging started.",
                      parts_indices_good.len());

        let timer_merging = clilog::stimer!("partition_merging");
        let effective_parts = process_partitions(
            &aig, staged, parts_indices_good, args.max_stage_degrad, args.stages
        ).unwrap();
        clilog::finish!(timer_merging);
        clilog::info!("after merging: {} parts.", effective_parts.len());
        effective_parts
    }).collect::<Vec<_>>();

    let f = std::fs::File::create(&args.parts_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &stages_effective_parts).unwrap();
}