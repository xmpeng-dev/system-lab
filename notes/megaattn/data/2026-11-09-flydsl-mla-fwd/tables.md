| B | S | proto bm128 khdef ms / TFLOPS | proto bm256 kh4 ms / TFLOPS | proto bm256 kh6 ms / TFLOPS | aiter ms / TFLOPS | te ms / TFLOPS | best proto / aiter | best proto / TE |
|---|---|---|---|---|---|---|---|---|
| 1 | 4K | 0.854 / 805 | 0.852 / 807 | 0.880 / 781 | 0.715 / 962 | 0.777 / 885 | 0.84 | 0.91 |
| 1 | 8K | 3.376 / 814 | 2.913 / 944 | 3.033 / 906 | 2.510 / 1095 | 2.695 / 1020 | 0.86 | 0.93 |
| 1 | 16K | 12.985 / 847 | 10.712 / 1026 | 11.214 / 981 | 9.321 / 1180 | 9.790 / 1123 | 0.87 | 0.91 |
| 1 | 32K | 51.096 / 861 | 40.992 / 1073 | 41.944 / 1049 | 36.314 / 1211 | 38.683 / 1137 | 0.89 | 0.94 |
| 1 | 64K | 202.958 / 867 | 160.997 / 1093 | 164.254 / 1071 | 143.273 / 1228 | 153.854 / 1143 | 0.89 | 0.96 |

geomean aiter: 0.869 (min 0.838, max 0.890, n 5)
geomean te: 0.930 (min 0.912, max 0.956, n 5)
geomean proto bm128 khdef: 0.741 (min 0.706, max 0.837, n 5)
geomean proto bm256 kh4: 0.869 (min 0.838, max 0.890, n 5)
geomean proto bm256 kh6: 0.842 (min 0.812, max 0.872, n 5)

same-GPU runs (proto, aiter, te back to back on one GPU):
| file | B | S | proto TFLOPS | aiter TFLOPS | TE TFLOPS | proto / aiter | proto / TE |
|---|---|---|---|---|---|---|---|
| bench_same_gpu0 | 1 | 4K | 798 | 963 | 886 | 0.829 | 0.901 |
| bench_same_gpu0 | 1 | 8K | 919 | 1123 | 1028 | 0.818 | 0.894 |
| bench_same_gpu0 | 1 | 16K | 988 | 1193 | 1125 | 0.828 | 0.878 |
| bench_same_gpu0 | 1 | 32K | 999 | 1212 | 1136 | 0.824 | 0.879 |
| bench_same_gpu0 | 1 | 64K | 1025 | 1226 | 1142 | 0.837 | 0.898 |
| bench_same_gpu1 | 1 | 4K | 817 | 1039 | 932 | 0.787 | 0.877 |
| bench_same_gpu1 | 1 | 8K | 942 | 1168 | 1069 | 0.807 | 0.881 |
| bench_same_gpu1 | 1 | 16K | 1028 | 1223 | 1146 | 0.840 | 0.897 |
| bench_same_gpu1 | 1 | 32K | 1076 | 1280 | 1205 | 0.840 | 0.893 |
| bench_same_gpu1 | 1 | 64K | 1083 | 1323 | 1232 | 0.818 | 0.879 |
| bench_same_gpu2_b2 | 2 | 4K | 821 | 1066 | 961 | 0.770 | 0.854 |
| bench_same_gpu2_b2 | 2 | 16K | 1023 | 1191 | 1106 | 0.859 | 0.925 |

same-GPU geomean proto / aiter B=1: 0.823 (min 0.787, max 0.840, n 10)
same-GPU geomean proto / aiter B=2: 0.813 (min 0.770, max 0.859, n 2)
same-GPU geomean proto / te B=1: 0.888 (min 0.877, max 0.901, n 10)
same-GPU geomean proto / te B=2: 0.889 (min 0.854, max 0.925, n 2)

check.jsonl:
| config | B | S | causal | O rel vs fp32 | O max abs vs fp32 | LSE max abs vs fp32 | O rel vs aiter | aiter O rel vs fp32 |
|---|---|---|---|---|---|---|---|---|
| bm128 khdef | 1 | 4K | 0 | 0.1518 | 0.0594 | 0.0373 | 0.1520 | 0.0023 |
| bm128 khdef | 1 | 16K | 0 | 0.0762 | 0.0197 | 0.0100 | 0.0766 | 0.0023 |
| bm128 khdef | 1 | 4K | 1 | 0.0020 | 0.0099 | 0.0053 | 0.0031 | 0.0017 |
| bm128 khdef | 1 | 8K | 1 | 0.0018 | 0.0113 | 0.0055 | 0.0032 | 0.0015 |
| bm128 khdef | 1 | 16K | 1 | 0.0017 | 0.0097 | 0.0048 | 0.0032 | 0.0015 |
| bm128 khdef | 1 | 32K | 1 | 0.0016 | 0.0091 | 0.0039 | 0.0032 | 0.0014 |
| bm128 khdef | 1 | 64K | 1 | 0.0016 | 0.0093 | 0.0045 | 0.0033 | 0.0013 |
| bm128 khdef | 2 | 4K | 1 | 0.0020 | 0.0103 | 0.0064 | 0.0031 | 0.0017 |
| bm128 khdef | 2 | 16K | 1 | 0.0018 | 0.0105 | 0.0041 | 0.0032 | 0.0015 |

check_bm256.jsonl:
| config | B | S | causal | O rel vs fp32 | O max abs vs fp32 | LSE max abs vs fp32 | O rel vs aiter | aiter O rel vs fp32 |
|---|---|---|---|---|---|---|---|---|
| bm256 kh4 | 1 | 4K | 1 | 0.0020 | 0.0099 | 0.0053 | 0.0031 | 0.0017 |
| bm256 kh4 | 1 | 8K | 1 | 0.0018 | 0.0113 | 0.0055 | 0.0032 | 0.0015 |
| bm256 kh4 | 1 | 16K | 1 | 0.0017 | 0.0097 | 0.0048 | 0.0032 | 0.0015 |
| bm256 kh4 | 1 | 32K | 1 | 0.0016 | 0.0091 | 0.0039 | 0.0032 | 0.0014 |
| bm256 kh4 | 1 | 64K | 1 | 0.0016 | 0.0093 | 0.0045 | 0.0033 | 0.0013 |
| bm256 kh4 | 2 | 4K | 1 | 0.0020 | 0.0103 | 0.0064 | 0.0031 | 0.0017 |
| bm256 kh4 | 2 | 16K | 1 | 0.0018 | 0.0105 | 0.0041 | 0.0032 | 0.0015 |

