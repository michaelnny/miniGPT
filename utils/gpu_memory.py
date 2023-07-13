# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

# Summary:
# the utility class Memory_Maximizer tracks reserved per epoch or per minibatch reserved GPU memory, in GB and as % of GPU VRAM,
# and most importantly programmatically confirms if any cudaMalloc retries took place.

# cudaMalloc retries can significantly lower performance (likely due to resetting the cache), but are otherwise
# not normally visible as an actual 'error' the way OOM is.

# usage - create instance,
# start() to reset internal stats, and begin,
# update() at end of epoch or minibatch,
# stop() to stop and print details.

# adjust batch size until you no longer see any cudaMalloc retries for best performance/memory maximization.

"""
example usage:

from gpu_memory import Memory_Maximizer

if rank == 0:
        memmax = Memory_Maximizer()

# memory and timing tracking
    if local_rank == 0:
        memmax.start()  # start will reset all tracking points

# in training loop - at minibatch or epoch end point:
    # update durations and memory tracking
    if local_rank == 0:
        memmax.update()

# at end of training - stop and print stats
    # memory summary
    if local_rank == 0:
        memmax.stop()  # stop and display info  
"""

import torch

gigabyte_size = 1073741824
megabyte_size = 1048576


def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


class Memory_Maximizer:
    def __init__(
        self,
    ):
        current_free, full_gpu_mem = torch.cuda.mem_get_info()

        self.m_total_gpu_memory = format_to_gb(full_gpu_mem)

        print(f"--> total memory per gpu (GB) = {self.m_total_gpu_memory}")

        self.m_reserved_memory_list = []
        self.m_reserved_memory_pct = []
        self.m_total_ooms = 0
        self.m_num_retries = 0
        self.m_max_reserved = 0

    def start(self):
        """start memory tracking, reset any current info"""

        torch.cuda.reset_peak_memory_stats()
        self.m_reserved_memory_list = []
        self.m_reserved_memory_pct = []
        self.m_num_retries = 0
        self.m_total_ooms = 0
        self.m_max_reserved = 0

        print(f"reserved and peak memory stats reset, ready to track")

    def update(
        self,
    ):
        """update reserved memory for this epoch"""
        updated_reserved = torch.cuda.memory_reserved()
        updated_reserved = format_to_gb(updated_reserved)

        self.m_reserved_memory_list.append(updated_reserved)
        self.m_reserved_memory_pct.append(round(100 * (updated_reserved / self.m_total_gpu_memory), 2))
        print(f"current epoch GPU memory = {self.m_reserved_memory_pct[-1]}%")

    def stop(
        self,
    ):
        """end of training...get various stats and display"""

        print(f"\nreserved memory = {self.m_reserved_memory_list}")
        print(f"memory % = {self.m_reserved_memory_pct}\n")

        cuda_max_reserved = format_to_gb(torch.cuda.max_memory_reserved())
        print(f"--> cuda max reserved memory = {cuda_max_reserved}")
        res_percentage = 100 * cuda_max_reserved / self.m_total_gpu_memory

        print(f"--> max reserved percentage = {round(res_percentage,4)}%\n")

        cuda_info = torch.cuda.memory_stats()

        self.m_num_retries = cuda_info.get("num_alloc_retries", 0)
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)

        print(f"cudaMalloc retries = {self.m_num_retries}")
        print(f"cuda OOM = {self.m_cuda_ooms}\n")
        if self.m_num_retries > 0:
            print(f"--> Recommend decreasing batch size...cuda retries can greatly degrade perf!")

    def summary(
        self,
    ):
        pass
