; Automatically converted to only require STRIPS and negative preconditions

(define (problem prob)
  (:domain spanner)
  (:objects bob spanner1 spanner2 spanner3 nut1 nut2 location1 location2 shed gate)
  (:init
    (at bob shed)
    (at spanner1 location1)
    (useable spanner1)
    (at spanner2 location1)
    (useable spanner2)
    (at spanner3 location2)
    (useable spanner3)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (link shed location1)
    (link location2 gate)
    (link location1 location2)
    (man bob)
    (locatable bob)
    (spanner spanner1)
    (locatable spanner1)
    (spanner spanner2)
    (locatable spanner2)
    (spanner spanner3)
    (locatable spanner3)
    (nut nut1)
    (locatable nut1)
    (nut nut2)
    (locatable nut2)
    (location location1)
    (location location2)
    (location shed)
    (location gate)
  )
  (:goal
    (and
      (tightened nut1)
      (tightened nut2)
    )
  )
)
