; Automatically converted to only require STRIPS and negative preconditions

(define (problem prob)
  (:domain spanner)
  (:objects bob spanner1 spanner2 spanner3 spanner4 spanner5 spanner6 nut1 nut2 nut3 nut4 nut5 nut6 location1 location2 location3 location4 location5 location6 location7 location8 location9 location10 shed gate)
  (:init
    (at bob shed)
    (at spanner1 location10)
    (useable spanner1)
    (at spanner2 location7)
    (useable spanner2)
    (at spanner3 location8)
    (useable spanner3)
    (at spanner4 location7)
    (useable spanner4)
    (at spanner5 location5)
    (useable spanner5)
    (at spanner6 location5)
    (useable spanner6)
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
    (link shed location1)
    (link location10 gate)
    (link location1 location2)
    (link location2 location3)
    (link location3 location4)
    (link location4 location5)
    (link location5 location6)
    (link location6 location7)
    (link location7 location8)
    (link location8 location9)
    (link location9 location10)
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
    )
  )
)
