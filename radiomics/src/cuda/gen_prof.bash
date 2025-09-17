#!/bin/bash

SHAPE_ARR=("ShapeKernelBasic" "ShapeKernelSharedMemory" "ShapeKernelSharedMemorySoa")
VOL_ARR=("VolumetryKernelBasic" "VolumetryKernelEqualWorkDistribution" "VolumetryKernelLocalAccumulatorWithAtomicFinal" "VolumetryKernelSoaMatrixBasedFullAtomics" "VolumetryKernelSoaMatrixBasedFullAtomicsReuseSameThreads" "VolumetryKernelSoaMatrixBasedAccumulatorsFinalAtomic" "VolumetryKernelBasicSoa" "VolumetryKernelSoaBlockReductionFinalAtomic")

mkdir -p prof_out

for item in "${SHAPE_ARR[@]}"; do

	echo "=================================================================="
	echo "         TESTING: ${item}"
	echo ""

	sudo ncu --set=full --export prof_out/rtx_4070_shape_${item} -f --kernel-name ${item} ./build/gcc-release/TEST_APP -f data/data_1 data/data_2 data/data_3 data/data_4 data/data_5 -r 1 --no-errors
done

for item in "${VOL_ARR[@]}"; do

	echo "=================================================================="
	echo "         TESTING: ${item}"
	echo ""

	sudo ncu --set=full --export prof_out/rtx_4070_volumetry_${item} -f --kernel-name ${item} ./build/gcc-release/TEST_APP -f data/data_1 data/data_2 data/data_3 data/data_4 data/data_5 -r 1 --no-errors
done

