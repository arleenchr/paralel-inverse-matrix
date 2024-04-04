# Tugas Kecil - Paralel Inverse Matrix

## Cara Kerja Paralelisasi Program
1. <b>Inisialiasi Matrix berdasarkan File Input</b>
<br>Matriks akan diinisialisasi berdasarkan file yang diinput. Matriks disimpan di variabel `inputMatrix`. Akan diinisialisasi juga matriks identitias.
<br><br>
2. <b>Iterasi untuk Setiap Baris pada Matriks</b>
<br>
2.1 <b>Menukar Baris dengan Baris Lainnya Jika Nilai Elemen Diagonal pada Baris Tersebut Nol</b>
<br> Jika terdapat baris yang elemen diagonalnya sama dengan nol, maka baris tersebut harus ditukar dengan baris lain. Jika tidak ada baris yang dapat ditukar, matriks tidak memiliki invers dan semua proses akan berhenti.
<br><br>
2.2 <b>Eliminasi elemen-elemen pada baris lain </b>
<br> Elemen-elemen pada baris lain dieliminasi dengan menggunakan <i>pivot factor</i> sehingga terbentuk matriks diagonal. Pada bagian ini, digunakan `pragma omp for` agar setiap iterasi terbagi/terparalelisasi ke semua <i>thread</i>.
<br><br>
2.3 <b>Membagi nilai elemen pada baris dengan nilai diagonal pada baris tersebut</b>
<br> Setiap elemen pada baris yang sedang diiterasi dibagi dengan nilai diagonalnya, sehingga terbentuk matriks diagonal yang elemennya berisikan angka 1 semua. Pada bagian ini, digunakan `pragma omp for` agar setiap iterasi terbagi/terparalelisasi ke semua
<br><br>
2.4 <b>Menampilkan hasil <i>inverse</i></b>
<br>Matriks hasil inverse berada pada matriks identitas yang diinisialisasi di awal. Matriks ini kemudian ditampilkan beserta dengan waktu eksekusi yang didapat dengan memanfaatkan `pragma omp barrier` dan fungsi `omp_get_wtime()`.

## How to Run
### Paralel Open-MP
Program inverse matrix paralel dengan Open MP ini dapat dijalankan dengan cara
<br> Contoh build, run, dan menyimpan output untuk test case `128.txt`

```console
user@user:~/kit-tucil-sister-2024$ make parallel-mp
user@user:~/kit-tucil-sister-2024$ make run-parallel-mp < test_cases/128.txt > bin/128-inversed.txt
```

### Serial

Contoh build, run, dan menyimpan output untuk test case `128.txt`.

```console
user@user:~/kit-tucil-sister-2024$ make serial
user@user:~/kit-tucil-sister-2024$ make run-serial < test_cases/128.txt > bin/128-inversed.txt
```