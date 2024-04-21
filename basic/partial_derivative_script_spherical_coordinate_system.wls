#!/usr/bin/env wolframscript
(* ::Package:: *)

h[r_, theta_, phi_, radius_, c_, beta_, i_] := (Sqrt[r ^ 2 + radius ^ 2 - 2 * r * radius * Sin[theta] * Cos[i * beta - phi]] - r) / c
FullSimplify[D[h[r, theta, phi, radius, c, beta, i], r]]
FullSimplify[D[h[r, theta, phi, radius, c, beta, i], theta]]
FullSimplify[D[h[r, theta, phi, radius, c, beta, i], phi]]



