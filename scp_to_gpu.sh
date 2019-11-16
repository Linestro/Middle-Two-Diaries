set -e
python mtx2coo.py > draft.cu
scp draft.cu xin.he@ray.seas.wustl.edu:/project/xzgroup-gpu/hexin/haojie/spmv.cu
echo 'Finishing scp spmv.cu'

