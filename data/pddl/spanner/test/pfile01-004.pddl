; Automatically converted to only require STRIPS and negative preconditions

(define (problem prob)
  (:domain spanner)
  (:objects bob spanner1 spanner2 spanner3 spanner4 spanner5 spanner6 spanner7 spanner8 spanner9 spanner10 spanner11 spanner12 spanner13 spanner14 spanner15 spanner16 spanner17 spanner18 spanner19 spanner20 spanner21 spanner22 spanner23 spanner24 spanner25 spanner26 spanner27 spanner28 spanner29 spanner30 spanner31 spanner32 spanner33 spanner34 spanner35 spanner36 spanner37 spanner38 spanner39 spanner40 spanner41 spanner42 spanner43 spanner44 spanner45 spanner46 spanner47 spanner48 spanner49 spanner50 spanner51 spanner52 spanner53 spanner54 spanner55 spanner56 spanner57 spanner58 spanner59 spanner60 spanner61 spanner62 spanner63 spanner64 spanner65 spanner66 spanner67 spanner68 spanner69 spanner70 spanner71 spanner72 spanner73 spanner74 spanner75 spanner76 spanner77 spanner78 spanner79 spanner80 spanner81 spanner82 spanner83 spanner84 spanner85 spanner86 spanner87 spanner88 spanner89 spanner90 spanner91 spanner92 spanner93 spanner94 spanner95 spanner96 spanner97 spanner98 spanner99 spanner100 nut1 nut2 nut3 nut4 nut5 nut6 nut7 nut8 nut9 nut10 nut11 nut12 nut13 nut14 nut15 nut16 nut17 nut18 nut19 nut20 nut21 nut22 nut23 nut24 nut25 nut26 nut27 nut28 nut29 nut30 nut31 nut32 nut33 nut34 nut35 nut36 nut37 nut38 nut39 nut40 nut41 nut42 nut43 nut44 nut45 nut46 nut47 nut48 nut49 nut50 nut51 nut52 nut53 nut54 nut55 nut56 nut57 nut58 nut59 nut60 nut61 nut62 nut63 nut64 nut65 nut66 nut67 nut68 nut69 nut70 nut71 nut72 nut73 nut74 nut75 nut76 nut77 nut78 nut79 nut80 nut81 nut82 nut83 nut84 nut85 nut86 nut87 nut88 nut89 nut90 nut91 nut92 nut93 nut94 nut95 nut96 nut97 nut98 nut99 nut100 location1 location2 location3 location4 location5 location6 location7 location8 location9 location10 location11 location12 location13 location14 location15 location16 location17 location18 location19 location20 location21 location22 location23 location24 location25 location26 location27 location28 location29 location30 location31 location32 location33 location34 location35 location36 location37 location38 location39 location40 location41 location42 location43 location44 location45 location46 location47 location48 location49 location50 shed gate)
  (:init
    (at bob shed)
    (at spanner1 location37)
    (useable spanner1)
    (at spanner2 location5)
    (useable spanner2)
    (at spanner3 location18)
    (useable spanner3)
    (at spanner4 location49)
    (useable spanner4)
    (at spanner5 location8)
    (useable spanner5)
    (at spanner6 location20)
    (useable spanner6)
    (at spanner7 location18)
    (useable spanner7)
    (at spanner8 location14)
    (useable spanner8)
    (at spanner9 location49)
    (useable spanner9)
    (at spanner10 location38)
    (useable spanner10)
    (at spanner11 location49)
    (useable spanner11)
    (at spanner12 location8)
    (useable spanner12)
    (at spanner13 location42)
    (useable spanner13)
    (at spanner14 location2)
    (useable spanner14)
    (at spanner15 location20)
    (useable spanner15)
    (at spanner16 location28)
    (useable spanner16)
    (at spanner17 location31)
    (useable spanner17)
    (at spanner18 location48)
    (useable spanner18)
    (at spanner19 location26)
    (useable spanner19)
    (at spanner20 location21)
    (useable spanner20)
    (at spanner21 location19)
    (useable spanner21)
    (at spanner22 location11)
    (useable spanner22)
    (at spanner23 location48)
    (useable spanner23)
    (at spanner24 location50)
    (useable spanner24)
    (at spanner25 location31)
    (useable spanner25)
    (at spanner26 location28)
    (useable spanner26)
    (at spanner27 location33)
    (useable spanner27)
    (at spanner28 location6)
    (useable spanner28)
    (at spanner29 location20)
    (useable spanner29)
    (at spanner30 location9)
    (useable spanner30)
    (at spanner31 location10)
    (useable spanner31)
    (at spanner32 location41)
    (useable spanner32)
    (at spanner33 location38)
    (useable spanner33)
    (at spanner34 location37)
    (useable spanner34)
    (at spanner35 location27)
    (useable spanner35)
    (at spanner36 location38)
    (useable spanner36)
    (at spanner37 location46)
    (useable spanner37)
    (at spanner38 location8)
    (useable spanner38)
    (at spanner39 location6)
    (useable spanner39)
    (at spanner40 location44)
    (useable spanner40)
    (at spanner41 location36)
    (useable spanner41)
    (at spanner42 location35)
    (useable spanner42)
    (at spanner43 location4)
    (useable spanner43)
    (at spanner44 location16)
    (useable spanner44)
    (at spanner45 location28)
    (useable spanner45)
    (at spanner46 location47)
    (useable spanner46)
    (at spanner47 location29)
    (useable spanner47)
    (at spanner48 location43)
    (useable spanner48)
    (at spanner49 location3)
    (useable spanner49)
    (at spanner50 location41)
    (useable spanner50)
    (at spanner51 location46)
    (useable spanner51)
    (at spanner52 location42)
    (useable spanner52)
    (at spanner53 location14)
    (useable spanner53)
    (at spanner54 location18)
    (useable spanner54)
    (at spanner55 location18)
    (useable spanner55)
    (at spanner56 location18)
    (useable spanner56)
    (at spanner57 location6)
    (useable spanner57)
    (at spanner58 location3)
    (useable spanner58)
    (at spanner59 location14)
    (useable spanner59)
    (at spanner60 location4)
    (useable spanner60)
    (at spanner61 location13)
    (useable spanner61)
    (at spanner62 location20)
    (useable spanner62)
    (at spanner63 location43)
    (useable spanner63)
    (at spanner64 location8)
    (useable spanner64)
    (at spanner65 location45)
    (useable spanner65)
    (at spanner66 location5)
    (useable spanner66)
    (at spanner67 location49)
    (useable spanner67)
    (at spanner68 location33)
    (useable spanner68)
    (at spanner69 location8)
    (useable spanner69)
    (at spanner70 location32)
    (useable spanner70)
    (at spanner71 location22)
    (useable spanner71)
    (at spanner72 location42)
    (useable spanner72)
    (at spanner73 location49)
    (useable spanner73)
    (at spanner74 location15)
    (useable spanner74)
    (at spanner75 location30)
    (useable spanner75)
    (at spanner76 location27)
    (useable spanner76)
    (at spanner77 location25)
    (useable spanner77)
    (at spanner78 location47)
    (useable spanner78)
    (at spanner79 location42)
    (useable spanner79)
    (at spanner80 location36)
    (useable spanner80)
    (at spanner81 location42)
    (useable spanner81)
    (at spanner82 location15)
    (useable spanner82)
    (at spanner83 location12)
    (useable spanner83)
    (at spanner84 location2)
    (useable spanner84)
    (at spanner85 location39)
    (useable spanner85)
    (at spanner86 location35)
    (useable spanner86)
    (at spanner87 location27)
    (useable spanner87)
    (at spanner88 location19)
    (useable spanner88)
    (at spanner89 location39)
    (useable spanner89)
    (at spanner90 location11)
    (useable spanner90)
    (at spanner91 location46)
    (useable spanner91)
    (at spanner92 location27)
    (useable spanner92)
    (at spanner93 location8)
    (useable spanner93)
    (at spanner94 location39)
    (useable spanner94)
    (at spanner95 location32)
    (useable spanner95)
    (at spanner96 location21)
    (useable spanner96)
    (at spanner97 location48)
    (useable spanner97)
    (at spanner98 location35)
    (useable spanner98)
    (at spanner99 location41)
    (useable spanner99)
    (at spanner100 location39)
    (useable spanner100)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (loose nut3)
    (at nut3 gate)
    (loose nut4)
    (at nut4 gate)
    (loose nut5)
    (at nut5 gate)
    (loose nut6)
    (at nut6 gate)
    (loose nut7)
    (at nut7 gate)
    (loose nut8)
    (at nut8 gate)
    (loose nut9)
    (at nut9 gate)
    (loose nut10)
    (at nut10 gate)
    (loose nut11)
    (at nut11 gate)
    (loose nut12)
    (at nut12 gate)
    (loose nut13)
    (at nut13 gate)
    (loose nut14)
    (at nut14 gate)
    (loose nut15)
    (at nut15 gate)
    (loose nut16)
    (at nut16 gate)
    (loose nut17)
    (at nut17 gate)
    (loose nut18)
    (at nut18 gate)
    (loose nut19)
    (at nut19 gate)
    (loose nut20)
    (at nut20 gate)
    (loose nut21)
    (at nut21 gate)
    (loose nut22)
    (at nut22 gate)
    (loose nut23)
    (at nut23 gate)
    (loose nut24)
    (at nut24 gate)
    (loose nut25)
    (at nut25 gate)
    (loose nut26)
    (at nut26 gate)
    (loose nut27)
    (at nut27 gate)
    (loose nut28)
    (at nut28 gate)
    (loose nut29)
    (at nut29 gate)
    (loose nut30)
    (at nut30 gate)
    (loose nut31)
    (at nut31 gate)
    (loose nut32)
    (at nut32 gate)
    (loose nut33)
    (at nut33 gate)
    (loose nut34)
    (at nut34 gate)
    (loose nut35)
    (at nut35 gate)
    (loose nut36)
    (at nut36 gate)
    (loose nut37)
    (at nut37 gate)
    (loose nut38)
    (at nut38 gate)
    (loose nut39)
    (at nut39 gate)
    (loose nut40)
    (at nut40 gate)
    (loose nut41)
    (at nut41 gate)
    (loose nut42)
    (at nut42 gate)
    (loose nut43)
    (at nut43 gate)
    (loose nut44)
    (at nut44 gate)
    (loose nut45)
    (at nut45 gate)
    (loose nut46)
    (at nut46 gate)
    (loose nut47)
    (at nut47 gate)
    (loose nut48)
    (at nut48 gate)
    (loose nut49)
    (at nut49 gate)
    (loose nut50)
    (at nut50 gate)
    (loose nut51)
    (at nut51 gate)
    (loose nut52)
    (at nut52 gate)
    (loose nut53)
    (at nut53 gate)
    (loose nut54)
    (at nut54 gate)
    (loose nut55)
    (at nut55 gate)
    (loose nut56)
    (at nut56 gate)
    (loose nut57)
    (at nut57 gate)
    (loose nut58)
    (at nut58 gate)
    (loose nut59)
    (at nut59 gate)
    (loose nut60)
    (at nut60 gate)
    (loose nut61)
    (at nut61 gate)
    (loose nut62)
    (at nut62 gate)
    (loose nut63)
    (at nut63 gate)
    (loose nut64)
    (at nut64 gate)
    (loose nut65)
    (at nut65 gate)
    (loose nut66)
    (at nut66 gate)
    (loose nut67)
    (at nut67 gate)
    (loose nut68)
    (at nut68 gate)
    (loose nut69)
    (at nut69 gate)
    (loose nut70)
    (at nut70 gate)
    (loose nut71)
    (at nut71 gate)
    (loose nut72)
    (at nut72 gate)
    (loose nut73)
    (at nut73 gate)
    (loose nut74)
    (at nut74 gate)
    (loose nut75)
    (at nut75 gate)
    (loose nut76)
    (at nut76 gate)
    (loose nut77)
    (at nut77 gate)
    (loose nut78)
    (at nut78 gate)
    (loose nut79)
    (at nut79 gate)
    (loose nut80)
    (at nut80 gate)
    (loose nut81)
    (at nut81 gate)
    (loose nut82)
    (at nut82 gate)
    (loose nut83)
    (at nut83 gate)
    (loose nut84)
    (at nut84 gate)
    (loose nut85)
    (at nut85 gate)
    (loose nut86)
    (at nut86 gate)
    (loose nut87)
    (at nut87 gate)
    (loose nut88)
    (at nut88 gate)
    (loose nut89)
    (at nut89 gate)
    (loose nut90)
    (at nut90 gate)
    (loose nut91)
    (at nut91 gate)
    (loose nut92)
    (at nut92 gate)
    (loose nut93)
    (at nut93 gate)
    (loose nut94)
    (at nut94 gate)
    (loose nut95)
    (at nut95 gate)
    (loose nut96)
    (at nut96 gate)
    (loose nut97)
    (at nut97 gate)
    (loose nut98)
    (at nut98 gate)
    (loose nut99)
    (at nut99 gate)
    (loose nut100)
    (at nut100 gate)
    (link shed location1)
    (link location50 gate)
    (link location1 location2)
    (link location2 location3)
    (link location3 location4)
    (link location4 location5)
    (link location5 location6)
    (link location6 location7)
    (link location7 location8)
    (link location8 location9)
    (link location9 location10)
    (link location10 location11)
    (link location11 location12)
    (link location12 location13)
    (link location13 location14)
    (link location14 location15)
    (link location15 location16)
    (link location16 location17)
    (link location17 location18)
    (link location18 location19)
    (link location19 location20)
    (link location20 location21)
    (link location21 location22)
    (link location22 location23)
    (link location23 location24)
    (link location24 location25)
    (link location25 location26)
    (link location26 location27)
    (link location27 location28)
    (link location28 location29)
    (link location29 location30)
    (link location30 location31)
    (link location31 location32)
    (link location32 location33)
    (link location33 location34)
    (link location34 location35)
    (link location35 location36)
    (link location36 location37)
    (link location37 location38)
    (link location38 location39)
    (link location39 location40)
    (link location40 location41)
    (link location41 location42)
    (link location42 location43)
    (link location43 location44)
    (link location44 location45)
    (link location45 location46)
    (link location46 location47)
    (link location47 location48)
    (link location48 location49)
    (link location49 location50)
    (man bob)
    (locatable bob)
    (spanner spanner1)
    (locatable spanner1)
    (spanner spanner2)
    (locatable spanner2)
    (spanner spanner3)
    (locatable spanner3)
    (spanner spanner4)
    (locatable spanner4)
    (spanner spanner5)
    (locatable spanner5)
    (spanner spanner6)
    (locatable spanner6)
    (spanner spanner7)
    (locatable spanner7)
    (spanner spanner8)
    (locatable spanner8)
    (spanner spanner9)
    (locatable spanner9)
    (spanner spanner10)
    (locatable spanner10)
    (spanner spanner11)
    (locatable spanner11)
    (spanner spanner12)
    (locatable spanner12)
    (spanner spanner13)
    (locatable spanner13)
    (spanner spanner14)
    (locatable spanner14)
    (spanner spanner15)
    (locatable spanner15)
    (spanner spanner16)
    (locatable spanner16)
    (spanner spanner17)
    (locatable spanner17)
    (spanner spanner18)
    (locatable spanner18)
    (spanner spanner19)
    (locatable spanner19)
    (spanner spanner20)
    (locatable spanner20)
    (spanner spanner21)
    (locatable spanner21)
    (spanner spanner22)
    (locatable spanner22)
    (spanner spanner23)
    (locatable spanner23)
    (spanner spanner24)
    (locatable spanner24)
    (spanner spanner25)
    (locatable spanner25)
    (spanner spanner26)
    (locatable spanner26)
    (spanner spanner27)
    (locatable spanner27)
    (spanner spanner28)
    (locatable spanner28)
    (spanner spanner29)
    (locatable spanner29)
    (spanner spanner30)
    (locatable spanner30)
    (spanner spanner31)
    (locatable spanner31)
    (spanner spanner32)
    (locatable spanner32)
    (spanner spanner33)
    (locatable spanner33)
    (spanner spanner34)
    (locatable spanner34)
    (spanner spanner35)
    (locatable spanner35)
    (spanner spanner36)
    (locatable spanner36)
    (spanner spanner37)
    (locatable spanner37)
    (spanner spanner38)
    (locatable spanner38)
    (spanner spanner39)
    (locatable spanner39)
    (spanner spanner40)
    (locatable spanner40)
    (spanner spanner41)
    (locatable spanner41)
    (spanner spanner42)
    (locatable spanner42)
    (spanner spanner43)
    (locatable spanner43)
    (spanner spanner44)
    (locatable spanner44)
    (spanner spanner45)
    (locatable spanner45)
    (spanner spanner46)
    (locatable spanner46)
    (spanner spanner47)
    (locatable spanner47)
    (spanner spanner48)
    (locatable spanner48)
    (spanner spanner49)
    (locatable spanner49)
    (spanner spanner50)
    (locatable spanner50)
    (spanner spanner51)
    (locatable spanner51)
    (spanner spanner52)
    (locatable spanner52)
    (spanner spanner53)
    (locatable spanner53)
    (spanner spanner54)
    (locatable spanner54)
    (spanner spanner55)
    (locatable spanner55)
    (spanner spanner56)
    (locatable spanner56)
    (spanner spanner57)
    (locatable spanner57)
    (spanner spanner58)
    (locatable spanner58)
    (spanner spanner59)
    (locatable spanner59)
    (spanner spanner60)
    (locatable spanner60)
    (spanner spanner61)
    (locatable spanner61)
    (spanner spanner62)
    (locatable spanner62)
    (spanner spanner63)
    (locatable spanner63)
    (spanner spanner64)
    (locatable spanner64)
    (spanner spanner65)
    (locatable spanner65)
    (spanner spanner66)
    (locatable spanner66)
    (spanner spanner67)
    (locatable spanner67)
    (spanner spanner68)
    (locatable spanner68)
    (spanner spanner69)
    (locatable spanner69)
    (spanner spanner70)
    (locatable spanner70)
    (spanner spanner71)
    (locatable spanner71)
    (spanner spanner72)
    (locatable spanner72)
    (spanner spanner73)
    (locatable spanner73)
    (spanner spanner74)
    (locatable spanner74)
    (spanner spanner75)
    (locatable spanner75)
    (spanner spanner76)
    (locatable spanner76)
    (spanner spanner77)
    (locatable spanner77)
    (spanner spanner78)
    (locatable spanner78)
    (spanner spanner79)
    (locatable spanner79)
    (spanner spanner80)
    (locatable spanner80)
    (spanner spanner81)
    (locatable spanner81)
    (spanner spanner82)
    (locatable spanner82)
    (spanner spanner83)
    (locatable spanner83)
    (spanner spanner84)
    (locatable spanner84)
    (spanner spanner85)
    (locatable spanner85)
    (spanner spanner86)
    (locatable spanner86)
    (spanner spanner87)
    (locatable spanner87)
    (spanner spanner88)
    (locatable spanner88)
    (spanner spanner89)
    (locatable spanner89)
    (spanner spanner90)
    (locatable spanner90)
    (spanner spanner91)
    (locatable spanner91)
    (spanner spanner92)
    (locatable spanner92)
    (spanner spanner93)
    (locatable spanner93)
    (spanner spanner94)
    (locatable spanner94)
    (spanner spanner95)
    (locatable spanner95)
    (spanner spanner96)
    (locatable spanner96)
    (spanner spanner97)
    (locatable spanner97)
    (spanner spanner98)
    (locatable spanner98)
    (spanner spanner99)
    (locatable spanner99)
    (spanner spanner100)
    (locatable spanner100)
    (nut nut1)
    (locatable nut1)
    (nut nut2)
    (locatable nut2)
    (nut nut3)
    (locatable nut3)
    (nut nut4)
    (locatable nut4)
    (nut nut5)
    (locatable nut5)
    (nut nut6)
    (locatable nut6)
    (nut nut7)
    (locatable nut7)
    (nut nut8)
    (locatable nut8)
    (nut nut9)
    (locatable nut9)
    (nut nut10)
    (locatable nut10)
    (nut nut11)
    (locatable nut11)
    (nut nut12)
    (locatable nut12)
    (nut nut13)
    (locatable nut13)
    (nut nut14)
    (locatable nut14)
    (nut nut15)
    (locatable nut15)
    (nut nut16)
    (locatable nut16)
    (nut nut17)
    (locatable nut17)
    (nut nut18)
    (locatable nut18)
    (nut nut19)
    (locatable nut19)
    (nut nut20)
    (locatable nut20)
    (nut nut21)
    (locatable nut21)
    (nut nut22)
    (locatable nut22)
    (nut nut23)
    (locatable nut23)
    (nut nut24)
    (locatable nut24)
    (nut nut25)
    (locatable nut25)
    (nut nut26)
    (locatable nut26)
    (nut nut27)
    (locatable nut27)
    (nut nut28)
    (locatable nut28)
    (nut nut29)
    (locatable nut29)
    (nut nut30)
    (locatable nut30)
    (nut nut31)
    (locatable nut31)
    (nut nut32)
    (locatable nut32)
    (nut nut33)
    (locatable nut33)
    (nut nut34)
    (locatable nut34)
    (nut nut35)
    (locatable nut35)
    (nut nut36)
    (locatable nut36)
    (nut nut37)
    (locatable nut37)
    (nut nut38)
    (locatable nut38)
    (nut nut39)
    (locatable nut39)
    (nut nut40)
    (locatable nut40)
    (nut nut41)
    (locatable nut41)
    (nut nut42)
    (locatable nut42)
    (nut nut43)
    (locatable nut43)
    (nut nut44)
    (locatable nut44)
    (nut nut45)
    (locatable nut45)
    (nut nut46)
    (locatable nut46)
    (nut nut47)
    (locatable nut47)
    (nut nut48)
    (locatable nut48)
    (nut nut49)
    (locatable nut49)
    (nut nut50)
    (locatable nut50)
    (nut nut51)
    (locatable nut51)
    (nut nut52)
    (locatable nut52)
    (nut nut53)
    (locatable nut53)
    (nut nut54)
    (locatable nut54)
    (nut nut55)
    (locatable nut55)
    (nut nut56)
    (locatable nut56)
    (nut nut57)
    (locatable nut57)
    (nut nut58)
    (locatable nut58)
    (nut nut59)
    (locatable nut59)
    (nut nut60)
    (locatable nut60)
    (nut nut61)
    (locatable nut61)
    (nut nut62)
    (locatable nut62)
    (nut nut63)
    (locatable nut63)
    (nut nut64)
    (locatable nut64)
    (nut nut65)
    (locatable nut65)
    (nut nut66)
    (locatable nut66)
    (nut nut67)
    (locatable nut67)
    (nut nut68)
    (locatable nut68)
    (nut nut69)
    (locatable nut69)
    (nut nut70)
    (locatable nut70)
    (nut nut71)
    (locatable nut71)
    (nut nut72)
    (locatable nut72)
    (nut nut73)
    (locatable nut73)
    (nut nut74)
    (locatable nut74)
    (nut nut75)
    (locatable nut75)
    (nut nut76)
    (locatable nut76)
    (nut nut77)
    (locatable nut77)
    (nut nut78)
    (locatable nut78)
    (nut nut79)
    (locatable nut79)
    (nut nut80)
    (locatable nut80)
    (nut nut81)
    (locatable nut81)
    (nut nut82)
    (locatable nut82)
    (nut nut83)
    (locatable nut83)
    (nut nut84)
    (locatable nut84)
    (nut nut85)
    (locatable nut85)
    (nut nut86)
    (locatable nut86)
    (nut nut87)
    (locatable nut87)
    (nut nut88)
    (locatable nut88)
    (nut nut89)
    (locatable nut89)
    (nut nut90)
    (locatable nut90)
    (nut nut91)
    (locatable nut91)
    (nut nut92)
    (locatable nut92)
    (nut nut93)
    (locatable nut93)
    (nut nut94)
    (locatable nut94)
    (nut nut95)
    (locatable nut95)
    (nut nut96)
    (locatable nut96)
    (nut nut97)
    (locatable nut97)
    (nut nut98)
    (locatable nut98)
    (nut nut99)
    (locatable nut99)
    (nut nut100)
    (locatable nut100)
    (location location1)
    (location location2)
    (location location3)
    (location location4)
    (location location5)
    (location location6)
    (location location7)
    (location location8)
    (location location9)
    (location location10)
    (location location11)
    (location location12)
    (location location13)
    (location location14)
    (location location15)
    (location location16)
    (location location17)
    (location location18)
    (location location19)
    (location location20)
    (location location21)
    (location location22)
    (location location23)
    (location location24)
    (location location25)
    (location location26)
    (location location27)
    (location location28)
    (location location29)
    (location location30)
    (location location31)
    (location location32)
    (location location33)
    (location location34)
    (location location35)
    (location location36)
    (location location37)
    (location location38)
    (location location39)
    (location location40)
    (location location41)
    (location location42)
    (location location43)
    (location location44)
    (location location45)
    (location location46)
    (location location47)
    (location location48)
    (location location49)
    (location location50)
    (location shed)
    (location gate)
  )
  (:goal
    (and
      (tightened nut1)
      (tightened nut2)
      (tightened nut3)
      (tightened nut4)
      (tightened nut5)
      (tightened nut6)
      (tightened nut7)
      (tightened nut8)
      (tightened nut9)
      (tightened nut10)
      (tightened nut11)
      (tightened nut12)
      (tightened nut13)
      (tightened nut14)
      (tightened nut15)
      (tightened nut16)
      (tightened nut17)
      (tightened nut18)
      (tightened nut19)
      (tightened nut20)
      (tightened nut21)
      (tightened nut22)
      (tightened nut23)
      (tightened nut24)
      (tightened nut25)
      (tightened nut26)
      (tightened nut27)
      (tightened nut28)
      (tightened nut29)
      (tightened nut30)
      (tightened nut31)
      (tightened nut32)
      (tightened nut33)
      (tightened nut34)
      (tightened nut35)
      (tightened nut36)
      (tightened nut37)
      (tightened nut38)
      (tightened nut39)
      (tightened nut40)
      (tightened nut41)
      (tightened nut42)
      (tightened nut43)
      (tightened nut44)
      (tightened nut45)
      (tightened nut46)
      (tightened nut47)
      (tightened nut48)
      (tightened nut49)
      (tightened nut50)
      (tightened nut51)
      (tightened nut52)
      (tightened nut53)
      (tightened nut54)
      (tightened nut55)
      (tightened nut56)
      (tightened nut57)
      (tightened nut58)
      (tightened nut59)
      (tightened nut60)
      (tightened nut61)
      (tightened nut62)
      (tightened nut63)
      (tightened nut64)
      (tightened nut65)
      (tightened nut66)
      (tightened nut67)
      (tightened nut68)
      (tightened nut69)
      (tightened nut70)
      (tightened nut71)
      (tightened nut72)
      (tightened nut73)
      (tightened nut74)
      (tightened nut75)
      (tightened nut76)
      (tightened nut77)
      (tightened nut78)
      (tightened nut79)
      (tightened nut80)
      (tightened nut81)
      (tightened nut82)
      (tightened nut83)
      (tightened nut84)
      (tightened nut85)
      (tightened nut86)
      (tightened nut87)
      (tightened nut88)
      (tightened nut89)
      (tightened nut90)
      (tightened nut91)
      (tightened nut92)
      (tightened nut93)
      (tightened nut94)
      (tightened nut95)
      (tightened nut96)
      (tightened nut97)
      (tightened nut98)
      (tightened nut99)
      (tightened nut100)
    )
  )
)
