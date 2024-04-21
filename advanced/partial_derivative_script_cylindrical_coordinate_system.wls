#!/usr/bin/env wolframscript
(* ::Package:: *)

h[rr_, zz_, ff_, radius_, c_, beta_, i_] := Sqrt[(rr * rr + radius * radius + zz * zz - 2 * rr * radius * Cos[i * beta - ff]) / (rr * rr + zz * zz)]
FullSimplify[D[h[rr, zz, ff, radius, c, beta, i], rr]]
FullSimplify[D[h[rr, zz, ff, radius, c, beta, i], zz]]
FullSimplify[D[h[rr, zz, ff, radius, c, beta, i], ff]]









