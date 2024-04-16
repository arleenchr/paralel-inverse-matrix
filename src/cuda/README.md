# Tugas Kecil - Paralel Inverse Matrix

## Cara Kerja Paralelisasi Program
1. <b>Inisialiasi Matrix berdasarkan File Input</b>
<br>Matriks akan diinisialisasi berdasarkan file yang diinput. Matriks disimpan di variabel `inputMatrix`. Akan diinisialisasi juga matriks identitias. Matriks yang tersimpan pada CPU disalin ke GPU dengan `cudaMalloc` dan `cudaMemcpy`.
<br><br>
2. <b>Inisialisasi ukuran <i>grid</i> dan <i>block</i> pada GPU</b>.
<br><i>Grid</i> yang diimplementasikan berukuran kolom `(size+16)/16` dan baris `(size+16)/16`, sementara <i>block</i> yang diimplementasikan berukuran kolom dan baris 16 x 16.
<br><br>
3. <b>Iterasi untuk Setiap Baris pada Matriks</b>
<br>
3.1 <b>Menukar Baris dengan Baris Lainnya Jika Nilai Elemen Diagonal pada Baris Tersebut Nol</b>
<br> Jika terdapat baris yang elemen diagonalnya sama dengan nol, maka baris tersebut harus ditukar dengan baris lain. Jika tidak ada baris yang dapat ditukar, matriks tidak memiliki invers dan semua proses akan berhenti.
<br><br>
3.2 <b>Eliminasi elemen-elemen pada baris lain </b>
<br> Elemen-elemen pada baris lain dieliminasi dengan menggunakan <i>pivot factor</i> sehingga terbentuk matriks diagonal. Eliminasi dilakukan pada GPU melalui fungsi global dengan ukuran <i>grid</i> dan <i>block</i> yang telah didefinisikan sebelumnya. Setelah eliminasi dilakukan, sinkronisasi dilakukan dengan `cudaDeviceSynchronize()`.
<br><br>
3.3 <b>Reduksi atau membagi nilai elemen pada baris dengan nilai diagonal pada baris tersebut</b>
<br> Setiap elemen pada baris yang sedang diiterasi dibagi dengan nilai diagonalnya, sehingga terbentuk matriks diagonal yang elemennya berisikan angka 1 semua. Reduksi dilakukan pada GPU melalui fungsi global dengan ukuran <i>grid</i> dan <i>block</i> yang telah didefinisikan sebelumnya. Setelah reduksi dilakukan, sinkronisasi dilakukan dengan `cudaDeviceSynchronize()`, lalu matriks pada GPU disalin ke matriks pada CPU.
<br><br>
3.4 <b>Menampilkan hasil <i>inverse</i></b>
<br>Matriks hasil inverse berada pada matriks identitas yang diinisialisasi di awal. Matriks ini kemudian ditampilkan beserta dengan waktu eksekusi yang didapat.

## How to Run
### Paralel CUDA
Program inverse matrix paralel dengan CUDA ini dapat dijalankan dengan cara membuka `cuda_colab.ipynb` pada Google Colab. Ganti <i>runtime type</i> pada Google Colab dengan T4 GPU. Lalu, <i>run all notebook</i>.

<br> Contoh build, run, dan menyimpan output untuk test case `128.txt`

```console
$ !nvcc cuda.cu matrix.c -o cuda
$ !./cuda < test_cases/128.txt > bin/238-inversed.txt
```

### Serial

Contoh build, run, dan menyimpan output untuk test case `128.txt`.

```console
user@user:~/if3230-tucil-cudalumping$ make serial
user@user:~/if3230-tucil-cudalumping$ make run-serial < test_cases/128.txt > bin/128-inversed.txt
```