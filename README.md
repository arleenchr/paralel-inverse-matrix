# Tugas Kecil - Paralel Inverse Matrix

## How to Run
### Paralel Open-MPI
Contoh build, run, dan menyimpan output untuk test case `128.txt`

```console
user@user:~/if3230-tucil-cudalumping$ make parallel-mpi
user@user:~/if3230-tucil-cudalumping$ make run-parallel-mpi < test_cases/128.txt > bin/128-inversed.txt
```

### Paralel Open-MP
Contoh build, run, dan menyimpan output untuk test case `128.txt`

```console
user@user:~/if3230-tucil-cudalumping$ make parallel-mp
user@user:~/if3230-tucil-cudalumping$ make run-parallel-mp < test_cases/128.txt > bin/128-inversed.txt
```

### Paralel CUDA
Program inverse matrix paralel dengan CUDA ini dapat dijalankan dengan cara membuka `cuda_colab.ipynb` pada Google Colab. Ganti <i>runtime type</i> pada Google Colab dengan T4 GPU. Lalu, <i>run all notebook</i>.

<br> Contoh build, run, dan menyimpan output untuk test case `128.txt`

```console
$ !nvcc cuda.cu matrix.c -o cuda
$ !./cuda < test_cases/128.txt > bin/238-inversed.txt
```
### Serial

Contoh build, run, dan menyimpan output untuk test case `32.txt`.

```console
user@user:~/kit-tucil-sister-2024$ make
user@user:~/kit-tucil-sister-2024$ cat ./test_cases/32.txt | ./bin/serial > 32.txt
```
