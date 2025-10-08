#!/bin/bash

# Create build log file with timestamp
BUILD_LOG="build_log_$(date +%Y%m%d_%H%M%S).txt"

echo "=== Build started at $(date) ===" | tee $BUILD_LOG
echo "Build log will be saved to: $BUILD_LOG" | tee -a $BUILD_LOG
echo "" | tee -a $BUILD_LOG

rm -rf build
mkdir build && cd build

echo "=== Running conan install ===" | tee -a ../$BUILD_LOG
conan install .. --build missing 2>&1 | tee -a ../$BUILD_LOG

echo "" | tee -a ../$BUILD_LOG
echo "=== Running cmake ===" | tee -a ../$BUILD_LOG
cmake .. 2>&1 | tee -a ../$BUILD_LOG

echo "" | tee -a ../$BUILD_LOG
echo "=== Running make ===" | tee -a ../$BUILD_LOG
make -j8 2>&1 | tee -a ../$BUILD_LOG

BUILD_STATUS=$?
echo "" | tee -a ../$BUILD_LOG
if [ $BUILD_STATUS -eq 0 ]; then
    echo "=== Build completed successfully at $(date) ===" | tee -a ../$BUILD_LOG
else
    echo "=== Build FAILED at $(date) with exit code $BUILD_STATUS ===" | tee -a ../$BUILD_LOG
    echo "Check $BUILD_LOG for details" | tee -a ../$BUILD_LOG
fi

exit $BUILD_STATUS