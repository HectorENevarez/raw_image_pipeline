#!/bin/bash

print_usage(){
	echo ""
	echo " Build the current project based."
	echo ""
	echo " Usage:"
	echo ""
	echo "  ./build.sh"
	echo "        Build project"
	echo ""
}

case "$1" in
	*)
		mkdir -p build
		cd build
		cmake ../
		make -j$(nproc)
		cd ../
		;;
esac
