# Tugas Kecil - Paralel Inverse Matrix


## Cara Kerja Paralelisasi Program
1. <b>Inisialiasi Matrix berdasarkan File Input</b>
<br>Proses ini dilakukan pada proses 0 saja. Ukuran matrix kemudian di-<i>broadcast</i> dengan fungsi `MPI_Bcast`. Di sini juga diinisialisi matriks output, yang berupa matriks identitas. Matriks ini yang nantinya akan menyimpan balikan (invers) dari matriks input.
<br><br>
2. <b>Iterasi untuk Setiap Baris pada Matriks</b>
<br>
2.1 <b>Menukar Baris dengan Baris Lainnya Jika Nilai Elemen Diagonal pada Baris Tersebut Nol</b>
<br> Jika terdapat baris yang elemen diagonalnya sama dengan nol, maka baris tersebut harus ditukar dengan baris lain. Jika tidak ada baris yang dapat ditukar, matriks tidak memiliki invers dan semua proses akan berhenti. Proses penukaran ini hanya dilakukan pada proses 0.
<br>
<br>
2.2 <b>Menyimpan dan Mem-<i>broadcast</i> nilai pivot ke semua proses</b> 
<br>Nilai Pivot adalah nilai yang berada pada diagonal matrix yang berkorespondensi terhadap baris tersebut. Nilai dikirim ke semua proses dengan fungsi `MPI_Bcast`.
<br><br>
2.3 <b>Membagi Setiap Elemen pada Baris dengan Nilai Pivot </b>
<br> Setiap elemen pada suatu baris harus dibagi dengan nilai pivot agar semua nilai diagonal matrix dapat diubah menjadi 1. Pada proses ini, elemen-elemen pada baris dibagi sama rata ke semua proses dengan fungsi `MPI_Scatter`. Hasil dari elemen-elemen yang sudah dibagi kemudian disatukan kembali dengan fungsi `MPI_Gather`.
<br><br>
2.4 <b>Mengirimkan Baris Pivot ke Semua Proses</b>
<br> Pada Operasi Baris Elementer (OBE), suatu baris memerlukan nilai dari baris pivot agar setiap nilai pada baris tersebut dapat dikurangi dan elemen yang berada pada kolom yang sama dengan nilai pivot dapat menjadi 0. Untuk itu, baris pivot akan dikirim ke semua proses dengan fungsi `MPI_Bcast`.
<br><br>
2.5 <b>Memecah Matriks Sama Rata ke Semua Proses</b>
<br>Matriks yang akan diproses akan dipecah ke semua proses yang berjalan dengan `MPI_Scatter` agar dapat dijalankan secara paralel (<i>data-level parallelism</i>)
<br><br>
2.6 <b>Operasi Baris Elementer</b>
<br>Dari setiap baris yang diproses, akan diambil faktor eliminasi. Kemudian, baris tersebut akan dikurangi dengan nilai pada pivot yang indeks kolomnya sama dengan nilai tersebut dikali dengan faktor eliminasi (<b>A[i][j] = P[j]*factor</b>, dengan A adalah matriks input dan P adalah baris pivot). Proses akan dilakukan sampai matriks input berubah menjadi matriks identitas.
<br><br>
2.7 <b> Menggabungkan Matriks </b>
<br> Baris-baris yang sudah diproses tadi kemudian digabungkan kembali dengan fungsi `MPI_Gather`. Program kemudian menampilkan matriks output yang menyimpan hasil matriks inverse dari matriks input.

## How to Run
### Paralel Open-MPI
Program inverse matrix paralel dengan Open MPI ini dapat dijalankan dengan cara
<br> Contoh build, run, dan menyimpan output untuk test case `128.txt`

```console
user@user:~/if3230-tucil-cudalumping$ make parallel-mpi
user@user:~/if3230-tucil-cudalumping$ make run-parallel-mpi < test_cases/128.txt > bin/128-inversed.txt
```

### Serial

Contoh build, run, dan menyimpan output untuk test case `128.txt`.

```console
user@user:~/if3230-tucil-cudalumping$ make serial
user@user:~/if3230-tucil-cudalumping$ make run-serial < test_cases/128.txt > bin/128-inversed.txt
```