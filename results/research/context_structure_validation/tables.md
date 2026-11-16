# Context / structure validation tables
| target | title | train n | test n | test MI | test p | first MI | first p | Cramér V | test JSD | log-loss gain | pattern MI | pattern p | degree-strat. p | structure gain beyond context | gain p | null gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20159 | GLS Audio Instrument Cable - Amp Cord for Bass & Electr | 1,348 | 1,887 | 0.0747 | 0.0020 | 0.0747 | 0.0020 | 0.112 | 0.041 | +0.0603 | 0.0928 | 0.0020 | 0.0020 | +0.0459 | 0.0050 | -0.0500 |
| 22372 | Donner Guitar Delay Pedal, Yellow Fall Analog Delay Gui | 1,087 | 1,536 | 0.0635 | 0.0040 | 0.0635 | 0.0020 | 0.113 | 0.036 | +0.0476 | 0.0771 | 0.0020 | 0.0020 | +0.0197 | 0.0050 | -0.0342 |
| 23516 | Ernie Ball Mondo Slinky Nickelwound Electric Guitar Str | 978 | 1,403 | 0.0628 | 0.0020 | 0.0628 | 0.0020 | 0.114 | 0.037 | +0.0471 | 0.1184 | 0.0020 | 0.0020 | +0.0317 | 0.0050 | -0.0535 |
| 23238 | D'Addario Guitar Strings - XL Nickel Electric Guitar St | 899 | 1,226 | 0.0739 | 0.0040 | 0.0739 | 0.0040 | 0.123 | 0.041 | +0.0550 | 0.1066 | 0.0020 | 0.0020 | +0.0475 | 0.0050 | -0.0384 |
| 23602 | D'Addario Guitar Strings - Phosphor Bronze Acoustic Gui | 760 | 1,078 | 0.0422 | 0.1138 | 0.0422 | 0.1038 | 0.117 | 0.027 | +0.0279 | 0.0815 | 0.0020 | 0.0040 | +0.0323 | 0.0050 | -0.0560 |
| 19314 | D'Addario Pro-Winder - Guitar String Winder, Guitar Str | 732 | 1,017 | 0.0350 | 0.6088 | 0.0350 | 0.6128 | 0.110 | 0.021 | +0.0236 | 0.0661 | 0.0020 | 0.0798 | +0.0416 | 0.0050 | -0.0505 |

## Does the named graphlet beat a simple triangle-count bin?

| target | graphlet gain | graphlet p | triangle-bin gain | triangle-bin p |
|---|---:|---:|---:|---:|
| 20159 | +0.0459 | 0.0050 | +0.0017 | 0.0050 |
| 22372 | +0.0197 | 0.0050 | +0.0014 | 0.0050 |
| 23516 | +0.0317 | 0.0050 | +0.0011 | 0.0050 |
| 23238 | +0.0475 | 0.0050 | +0.0019 | 0.0050 |
| 23602 | +0.0323 | 0.0050 | +0.0020 | 0.0050 |
| 19314 | +0.0416 | 0.0050 | +0.0017 | 0.0050 |

## Context signatures: target 20159

| context | n train | n test | top predecessors | top next | pattern share | mean triangles | mean pred degree |
|---|---:|---:|---|---|---|---:|---:|
| C0 | 155 | 206 | Amazon Basics Adjustable Guitar Folding A-; Guitar Capo (2 Pack) for Guitars, Ukulele,; Nordic Essentials Guitar Capo Deluxe with ; ChromaCast Acoustic Mini Jumbo Padded Gig  | GLS Audio Speaker Cable 1/4" to 1/4" - 12 ; Guitar Savers Premium Strap Locks (2 Pair); Korg TM50BK Instrument Tuner and Metronome; Gator Frameworks Adjustable Double Stand H | none:81, multi:40, K2_3:13, K4_tail:10, K5:6, T5:4 | 67.90322580645162 | 175.3 |
| C1 | 63 | 100 | Snark ST-2 Multi-Instrument Chromatic Tune; Snark ST-2 All Instrument Clip-On Chromati; IK Multimedia iRig 2 Portable Guitar Audio; Xvive U4 Wireless in-Ear Monitor System Pr | Donner DPA-1 Guitar Pedal Power Supply Ada; Talent iCLaw Mic/Music Stand Holder for iP; String Swing Metal Guitar Wall Mount for C; XTUGA RW2080 in Ear Monitor System 2 Chann | none:43, multi:8, K2_3:5, K4_tail:5, C5:2 | 44.42857142857143 | 130.7 |
| C2 | 51 | 71 | Dunlop 5010 7" Mic Stand Pickholders-Black; Shure SM57 Dynamic Instrument Mic - Profes; Mic Cable Clips - Universal Microphone Boo; On-Stage MS7701B Tripod Microphone Boom St | Grindhouse Speakers - GH10M - 10 Inch Pass; Vizcaya Violin Stand VLH10 Violin Hanger W; Paiste PST 7 Cymbal (PAISTE-PST7-Splash10); D'Addario Guitar Strings - XL Nickel Elect | none:36, multi:6, C5:4, K5:3, K4_tail:2 | 46.627450980392155 | 142.6 |
| C5 | 110 | 157 | KLIQ UberTuner-Max - Rechargable Professio; D'Addario Pro-Winder - Guitar String Winde; iLuiz Guitar Neck Notched Straight Edge Lu; Jim Dunlop System 65 Guitar Maintenance Ki | D'Addario Guitar Strings - XL Nickel Elect; KLIQ TinyTune Pro Stage Tuner Pedal for Gu; Amazon Basics Adjustable Guitar Folding A-; Donner Guitar Delay Pedal, Yellow Fall Ana | none:57, multi:28, K2_3:14, K5:7, C5:2, K4_tail:2 | 79.10909090909091 | 213.3 |
| C6 | 137 | 186 | GLS Audio Speaker Cable 1/4" to 1/4" - 12 ; GLS Audio 12ft Mic Cable Patch Cords - XLR; GLS Audio 6 Inch Patch Cable Cords (0.5 fe; Amazon Basics 2-Pack XLR Microphone Cable  | Ernie Ball Mondo Slinky Nickelwound Electr; Boss RC-3 Loop Station Guitar Effects Peda; GLS Audio 6 Inch Patch Cable Cords (0.5 fe; D'Addario Guitar Strings - XL Nickel Elect | none:64, multi:44, K2_3:11, T5:5, K4_tail:5, C5:4 | 91.33576642335767 | 286.1 |
| C7 | 117 | 148 | Blackstar Fly 3 Limited Edition Mini Ampli; Boss SD-1 Super Overdrive Pedal; VOX AP2BS amPlug 2 Guitar/Bass Headphone A; J. Rockett Audio Designs Tour Series Lenny | Hola! Music HM-MTH Microphone Music Stand ; Donner Small Guitar Pedal Board, Mini Effe; On Stage XCG4 Velveteen Padded Tubular Gui; Korg BA40 Bluegrass Tuner for Mandolin, Fi | none:80, K2_3:15, multi:11, K4_tail:6, K5:5 | 41.0940170940171 | 103.9 |
| C9 | 88 | 120 | Hola! Music Guitar Stand - Height Adjustab; Gator Cases Padded Nylon Gear Carry Remova; Gearlux Die-Cast Music Sheet Stand with Hy; TMS Guitar Stand 9 Holder Guitar Folding S | Top Stage Pro Universal Guitar Stand; Levy's Leathers Suede Guitar Strap with Sh; Clip On Guitar Tuner For All Instruments, ; Antrader 90 Degree Right Angle 1/4" TS Mon | none:43, multi:20, K5:10, K2_3:9, K4_tail:5, C5:1 | 59.93181818181818 | 164.7 |
| C10 | 80 | 109 | Fender Premium Celluloid Guitar Picks 351 ; Jim Dunlop Pick Holder, Black, 3/Bag; Dunlop Pick Variety Pack, Assorted, Light/; JIM DUNLOP 449P.73 Max-Grip® Nylon Standar | Fender Premium Celluloid Guitar Picks 351 ; Snark ST-2 Multi-Instrument Chromatic Tune; Martin Marquis 80/20 Bronze Acoustic Guita; GLS Audio 12ft Mic Cable Patch Cords - XLR | none:32, multi:29, K2_3:9, K5:7, K1_4:1, K4_tail:1 | 87.375 | 210.1 |
| C13 | 92 | 122 | Donner Guitar Delay Pedal, Yellow Fall Ana; DADDARIO 9-Volt Power Adaptor (PW-CT-9V); Donner DPA-1 Guitar Pedal Power Supply Ada; Donner DP-1 Guitar Power Supply 10 Isolate | JOYO Zombie-II (Dual Rectifier) BanTamp XL; Ernie Ball Mondo Slinky Nickelwound Electr; Guitar Capo (2 Pack) for Guitars, Ukulele,; Donner DPA-1 Guitar Pedal Power Supply Ada | multi:36, none:31, K2_3:12, K4_tail:9, K5:3, C5:1 | 133.6413043478261 | 388.9 |
| C15 | 66 | 88 | Snark SN-5 Tuner for Guitar, Bass and Viol; Behringer TM300 Tube Amp Modeler Pedal; Snark SA-2 5 Pedal Daisy Chain; Walker & Williams F-01 Classic Black Gator | Snark SN-5 Tuner for Guitar, Bass and Viol; Firefly FFTH Semi-Hollow body Guitar Orang; CAHAYA 5 in 1 Heightened Dual-use Sheet Mu; SKB Bi-Fold Wallet (1SKB-44PRO) | none:46, multi:12, K2_3:6, K5:2 | 72.93939393939394 | 205.7 |
| C17 | 153 | 241 | D'Addario Guitar Strings - Phosphor Bronze; Ernie Ball Mondo Slinky Nickelwound Electr; Guitar Savers Premium Strap Locks (2 Pair); Ernie Ball Polypro Guitar Strap, Olive (P0 | Elixir® Strings 16545 Acoustic Phosphor Br; Fender Premium Celluloid Guitar Picks 351 ; GLS Audio Speaker Cable 1/4" to 1/4" - 12 ; Guitar Savers Premium Strap Locks (2 Pair) | none:81, multi:50, K5:11, K2_3:6, K4_tail:4, C5:1 | 111.93464052287581 | 305.7 |
| C18 | 77 | 102 | Snark SN5X Clip-On Tuner for Guitar, Bass ; Fender Strap Blocks (2 Pair); JIM DUNLOP Curved, Nickel, Guitar Capo; Fender Deluxe Locking Staggered Guitar Tun | Guitar Savers Premium Strap Locks (2 Pair); Twinote Pπ FUZZ Analog Modern Fuzz Guitar ; Wolfride 4 PCS Guitar Strap Locks and Butt; Zoom G5n Guitar Multi-Effects Processor wi | none:35, multi:28, K2_3:10, K4_tail:3, C5:1 | 114.68831168831169 | 306.2 |

## Largest context contrasts: target 20159

| context A | n A | context B | n B | JSD | A top next | B top next |
|---|---:|---|---:|---:|---|---|
| C1 | 63 | C18 | 77 | 0.130 | Donner DPA-1 Guitar Pedal Power Supply; Talent iCLaw Mic/Music Stand Holder fo; String Swing Metal Guitar Wall Mount f; XTUGA RW2080 in Ear Monitor System 2 C | Guitar Savers Premium Strap Locks (2 P; Twinote Pπ FUZZ Analog Modern Fuzz Gui; Wolfride 4 PCS Guitar Strap Locks and ; Zoom G5n Guitar Multi-Effects Processo |
| C1 | 63 | C9 | 88 | 0.115 | Donner DPA-1 Guitar Pedal Power Supply; Talent iCLaw Mic/Music Stand Holder fo; String Swing Metal Guitar Wall Mount f; XTUGA RW2080 in Ear Monitor System 2 C | Top Stage Pro Universal Guitar Stand; Levy's Leathers Suede Guitar Strap wit; Clip On Guitar Tuner For All Instrumen; Antrader 90 Degree Right Angle 1/4" TS |
| C1 | 63 | C17 | 153 | 0.113 | Donner DPA-1 Guitar Pedal Power Supply; Talent iCLaw Mic/Music Stand Holder fo; String Swing Metal Guitar Wall Mount f; XTUGA RW2080 in Ear Monitor System 2 C | Elixir® Strings 16545 Acoustic Phospho; Fender Premium Celluloid Guitar Picks ; GLS Audio Speaker Cable 1/4" to 1/4" -; Guitar Savers Premium Strap Locks (2 P |
| C2 | 51 | C18 | 77 | 0.113 | Grindhouse Speakers - GH10M - 10 Inch ; Vizcaya Violin Stand VLH10 Violin Hang; Paiste PST 7 Cymbal (PAISTE-PST7-Splas; D'Addario Guitar Strings - XL Nickel E | Guitar Savers Premium Strap Locks (2 P; Twinote Pπ FUZZ Analog Modern Fuzz Gui; Wolfride 4 PCS Guitar Strap Locks and ; Zoom G5n Guitar Multi-Effects Processo |
| C1 | 63 | C15 | 66 | 0.112 | Donner DPA-1 Guitar Pedal Power Supply; Talent iCLaw Mic/Music Stand Holder fo; String Swing Metal Guitar Wall Mount f; XTUGA RW2080 in Ear Monitor System 2 C | Snark SN-5 Tuner for Guitar, Bass and ; Firefly FFTH Semi-Hollow body Guitar O; CAHAYA 5 in 1 Heightened Dual-use Shee; SKB Bi-Fold Wallet (1SKB-44PRO) |
| C0 | 155 | C1 | 63 | 0.105 | GLS Audio Speaker Cable 1/4" to 1/4" -; Guitar Savers Premium Strap Locks (2 P; Korg TM50BK Instrument Tuner and Metro; Gator Frameworks Adjustable Double Sta | Donner DPA-1 Guitar Pedal Power Supply; Talent iCLaw Mic/Music Stand Holder fo; String Swing Metal Guitar Wall Mount f; XTUGA RW2080 in Ear Monitor System 2 C |

## Structure signatures: target 20159

| pattern | n train | top contexts | top next | mean triangles |
|---|---:|---|---|---:|
| C5 | 22 | C6:4, C-1:4, C2:4 | DunlopJohn Petrucci Jazz III, 6/Player's P; Korg BA40 Bluegrass Tuner for Mandolin, Fi; Behringer TM300 Tube Amp Modeler Pedal | 33.50 |
| K2_3 | 127 | C-1:17, C7:15, C5:14 | Donner Guitar Delay Pedal, Yellow Fall Ana; D'Addario Pro-Winder - Guitar String Winde; Electro-Harmonix NIC10 Nickel Wound Light  | 41.43 |
| K4_tail | 57 | C0:10, C13:9, C7:6 | Ukulele Concert Size Bundle From Lohanu (L; GLS Audio Speaker Cable 1/4" to 1/4" - 12 ; M-Audio SP-2 - Universal Sustain Pedal wit | 56.93 |
| K5 | 59 | C17:11, C9:10, C5:7 | Guitar Savers Premium Strap Locks (2 Pair); Elixir® Strings 16545 Acoustic Phosphor Br; Tension Springs for Tremolo Arms (4 pieces | 96.97 |
| multi | 320 | C17:50, C6:44, C0:40 | Ernie Ball Mondo Slinky Nickelwound Electr; Donner DP-1 Guitar Power Supply 10 Isolate; Donner DC-2 Guitar Capo for Electric and A | 211.34 |
| none | 752 | C-1:123, C17:81, C0:81 | On Stage XCG4 Velveteen Padded Tubular Gui; D'Addario Guitar Strings - XL Nickel Elect; Ernie Ball Mondo Slinky Nickelwound Electr | 22.68 |

## Pooled structure signatures (all targets, TRAIN)

| pattern | n | targets | mean triangles | top contexts | top next |
|---|---:|---:|---:|---|---|
| C5 | 100 | 6 | 26.8 | C5:11, C2:10, C7:9, C17:9 | Ahead Drumsticks (LT); Moukey Dynamic Microphone, Karaoke Microph; Fender Standard Vintage Guitar Strap, Blac; Hiree Guitar Pedal Power Supply Adapter St; Ernie Ball Beefy Slinky Cobalt Electric Gu |
| K1_4 | 5 | 4 | 13.0 | C8:1, C10:1, C2:1, C13:1 | KIRLIN Cable IWB-202BFGL-20/BR 20-Feet Pre; Glory Black Laquer E Flat Alto Saxophone w; Behringer XM8500; Hercules Microphone Stand (MH100B); Blackstar ID30 Programmable Combo with Eff |
| K2_3 | 374 | 6 | 35.9 | C7:52, C0:36, C13:35, C5:34 | Donner Fuzz Pedal, Stylish Fuzz Guitar Ped; Dunlop D38 Classic Woven Guitar and Bass S; D'Addario Pro-Winder - Guitar String Winde; D'Addario Guitar Strings - XL Nickel Elect; D'Addario Accessories Guitar Tuner - Micro |
| K4_tail | 245 | 6 | 52.9 | C17:34, C0:30, C5:27, C7:25 | Ernie Ball Mondo Slinky Nickelwound Electr; Snark ST-2 All Instrument Clip-On Chromati; D'Addario Pro-Winder - Guitar String Winde; Donner DPA-1 Guitar Pedal Power Supply Ada; Ernie Ball Hybrid Slinky Classic Pure Nick |
| K5 | 346 | 6 | 104.3 | C17:82, C10:50, C5:46, C0:30 | On Stage XCG4 Velveteen Padded Tubular Gui; D'Addario Pro-Winder - Guitar String Winde; D'Addario Guitar Strings - XL Nickel Elect; Ernie Ball Mondo Slinky Nickelwound Electr; Snark SN-5 Tuner for Guitar, Bass and Viol |
| T5 | 15 | 5 | 62.0 | C0:7, C6:5, C7:1, C5:1 | Fender Mustang I Amplifier Cover; Gavitt Guitar Wire Cloth-Covered Vintage-S; ESP LTD EC-256 Electric Guitar, Black Sati; GHS Strings A260 Bobby Osborne Signature S; La Bella 750N-B Black Nylon Tape, 5-String |
| double_triangle | 10 | 4 | 8.7 | C13:2, C5:2, C15:2, C7:1 | Evans EMAD Clear Tom Hoop Drum Head, 16 In; Tascam DP-32SD 32-Track Digital Portastudi; Audio2000'S In- Ear Audio Monitor System (; Sound Addicted - Corner Bass Traps (4 Pack; Vintage Forge Black Curved Bottom Humbucke |
| multi | 1,472 | 6 | 189.3 | C17:401, C5:212, C10:164, C6:145 | D'Addario Guitar Strings - Phosphor Bronze; Snark SN-5 Tuner for Guitar, Bass and Viol; Snark SN1 Guitar Tuner (Blue); Ernie Ball Mondo Slinky Nickelwound Electr; D'Addario Celluloid Guitar Picks - 10 Pack |
| none | 3,237 | 6 | 23.8 | C7:376, C0:343, C17:343, C5:295 | Ernie Ball Mondo Slinky Nickelwound Electr; D'Addario Guitar Strings - Phosphor Bronze; Donner DPA-1 Guitar Pedal Power Supply Ada; D'Addario Guitar Strings - XL Nickel Elect; Snark SN-5 Tuner for Guitar, Bass and Viol |

