# Prototip: stratul simbolic (SATB)

Nu e încă legat la pipeline. Rulează de sine stătător, cu `music21` instalat:

    pip install music21

## Ce e aici

| fișier | ce face |
|---|---|
| `satb/rules.py` | regulile de voice leading ca verificatoare: registre, spațiere, încrucișare, cvinte/octave paralele, salturi |
| `satb/calibrate.py` | rulează regulile peste coralele Bach din corpusul `music21` — calibrarea |
| `satb/harmonize.py` | armonizatorul: Viterbi exact peste voicing-uri |
| `satb/legal_vs_good.py` | arată că „fără încălcări" nu înseamnă „bun" |
| `satb/style_prior.py` | adaugă prior învățat din corpus |
| `satb/singability.py` | al doilea scor: cât de cântabilă e linia |
| `satb/integrate.py` | lanțul complet: SATB generat → timbru învățat → stem audio |

## Observația structurală

Toate constrângerile dure sunt fie *în interiorul* unui acord (registre,
spațiere, note din acord), fie *între două acorduri vecine* (paralele,
salturi). Niciuna nu se întinde mai departe de un acord înapoi.

Un graf de constrângeri ale cărui muchii leagă doar vecini e un lanț, iar un
lanț se rezolvă exact prin programare dinamică. Fără euristici, fără
backtracking, fără model. Zero încălcări nu e un rezultat bun — e o garanție,
fiindcă tranzițiile ilegale lipsesc din graf.

## Ce s-a măsurat

Regulile, pe 60 de corale Bach (4778 evenimente), încălcări la 100 de
evenimente:

    octave paralele    0.02   <- practic absolută
    cvinte paralele    0.17   <- practic absolută
    registre           0.33
    salturi            0.67
    spațiere           1.40
    încrucișare        3.03   <- NU e regulă dură; A-T e locul obișnuit

Partidele sunt corect atribuite (verificat: nume Soprano/Alto/Tenor/Bass,
înălțimi medii 71.0 / 65.5 / 59.8 / 51.5, descrescător curat).

Armonizatorul pe bucle de laudă: 0 încălcări, 5–580 ms, optim global.

Profil de scriitură (medii, semitonuri):

    Bach                    sopran 70.9   S-A 5.4   A-T 5.7   T-B 8.4
    generat, fără prior     sopran 66.8   S-A 6.0   A-T 9.8   T-B 6.5
    generat, cu prior       sopran 72.8   S-A 5.2   A-T 4.5   T-B 9.0

Cu prior, profilul se apropie de corpus **și** încălcările rămân 0. Cele două
straturi se compun: constrângerile dau corectitudinea, corpusul dă gustul.

Cântabilitatea prinde ce profilul de spațiere ratează:

              grad alăturat   salturi >cvartă   în extreme   static
    Bach  T             79%              6%           5%         38%
          B             64%             16%          11%         23%
    gen   T            100%              0%          50%         53%
          B             67%              0%          50%         20%

Tenorul și basul stau jumătate din timp la marginea registrului.

## Două lucruri de ținut minte

**Bach e corpusul corect de validare și corpusul greșit de stil.** Cele 433 de
corale verifică mașinăria. Dar toate cifrele de calitate de mai sus măsoară
*apropierea de Bach*, iar un cor penticostal românesc nu cântă ca Bach. Un
prior derivat din Bach dă SATB corect cu sonoritate de corală luterană. Cele
124 de partituri ale voastre **sunt** stilul; Bach doar verifică motorul.

**Orice componentă trebuie să aibă o ieșire „nu știu".** `music21` dă cifră
romană la 100% din acorduri — asta e acoperire, nu corectitudine: nu refuză
niciodată. La fel, `harmonize()` întoarce `None` fără să spună dacă melodia
avea o notă din afara acordului sau dacă nu exista tranziție legală. De
reparat înainte de portare.

## Ce urmează

1. Portat în repo ca modul, cu ieșire „nu știu"
2. Format de intrare pentru acorduri, legat de grila de bătăi din `analyze()`
3. Prior din cele 124 de partituri (blocat până există ca MusicXML)
4. Stare extinsă (ultimele două voicing-uri) pentru forma frazei
5. Export MusicXML per instrument, în pasul de export
