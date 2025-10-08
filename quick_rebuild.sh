#!/bin/bash
echo "Quick rebuild..."
cd /home/cc/NeuPIMs/build
make -j8 2>&1 | tail -20
echo ""
if [ $? -eq 0 ]; then
    echo "✅ Build successful! Run: ./brun.sh"
else
    echo "❌ Build failed!"
fi

