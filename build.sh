#!/bin/bash

print_usage(){
	echo ""
	echo " Build the current project based."
	echo ""
	echo " Usage:"
	echo ""
	echo "  ./build.sh cpu"
	echo "        Build with no GPU accelerators"
	echo ""
	echo "  ./build.sh cuda"
	echo "        Build with cuda enabled"
	echo ""
	echo ""
}

case "$1" in
	cpu)
		mkdir -p build
		cd build
		cmake ../
		make -j$(nproc)
		cd ../
		;;
	cuda)
		mkdir -p build
		cd build
		cmake -DENABLE_CUDA=true ../
		make -j$(nproc)
		cd ../
		;;

	native)
		mkdir -p build
		cd build
		cmake ${EXTRA_OPTS} ../
		make -j$(nproc)
		cd ../
		;;

	*)
		print_usage
		exit 1
		;;
esac


mkdir -p build
cd build
cmake ../
make -j$(nproc)
cd ../
