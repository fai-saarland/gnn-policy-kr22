; Automatically converted to only require STRIPS and negative preconditions

(define (problem delivery-5x5-3)
  (:domain delivery)
  (:objects c_0_0 c_0_1 c_0_2 c_0_3 c_0_4 c_1_0 c_1_1 c_1_2 c_1_3 c_1_4 c_2_0 c_2_1 c_2_2 c_2_3 c_2_4 c_3_0 c_3_1 c_3_2 c_3_3 c_3_4 c_4_0 c_4_1 c_4_2 c_4_3 c_4_4 p1 p2 p3 t1)
  (:init
    (adjacent c_0_2 c_1_2)
    (adjacent c_3_3 c_4_3)
    (adjacent c_1_3 c_2_3)
    (adjacent c_3_3 c_3_2)
    (adjacent c_0_3 c_0_2)
    (adjacent c_2_0 c_3_0)
    (adjacent c_3_1 c_2_1)
    (adjacent c_0_3 c_0_4)
    (adjacent c_1_4 c_1_3)
    (adjacent c_2_2 c_2_3)
    (adjacent c_3_2 c_3_3)
    (adjacent c_4_0 c_4_1)
    (adjacent c_0_2 c_0_3)
    (adjacent c_3_2 c_2_2)
    (adjacent c_3_1 c_4_1)
    (adjacent c_4_1 c_3_1)
    (adjacent c_2_4 c_2_3)
    (adjacent c_1_1 c_1_0)
    (adjacent c_4_3 c_4_2)
    (adjacent c_4_4 c_4_3)
    (adjacent c_0_1 c_0_2)
    (adjacent c_2_4 c_3_4)
    (adjacent c_1_1 c_0_1)
    (adjacent c_2_3 c_3_3)
    (adjacent c_4_2 c_4_3)
    (adjacent c_0_3 c_1_3)
    (adjacent c_1_4 c_0_4)
    (adjacent c_1_3 c_1_2)
    (adjacent c_3_0 c_2_0)
    (adjacent c_4_3 c_4_4)
    (adjacent c_0_0 c_1_0)
    (adjacent c_2_1 c_2_2)
    (adjacent c_2_3 c_2_4)
    (adjacent c_0_0 c_0_1)
    (adjacent c_2_2 c_1_2)
    (adjacent c_1_4 c_2_4)
    (adjacent c_0_1 c_0_0)
    (adjacent c_1_0 c_1_1)
    (adjacent c_2_1 c_2_0)
    (adjacent c_4_0 c_3_0)
    (adjacent c_1_2 c_1_1)
    (adjacent c_1_3 c_1_4)
    (adjacent c_4_2 c_3_2)
    (adjacent c_2_3 c_1_3)
    (adjacent c_3_0 c_4_0)
    (adjacent c_4_4 c_3_4)
    (adjacent c_4_1 c_4_0)
    (adjacent c_1_1 c_2_1)
    (adjacent c_3_0 c_3_1)
    (adjacent c_1_1 c_1_2)
    (adjacent c_3_2 c_4_2)
    (adjacent c_2_4 c_1_4)
    (adjacent c_2_3 c_2_2)
    (adjacent c_3_4 c_3_3)
    (adjacent c_0_4 c_0_3)
    (adjacent c_1_0 c_0_0)
    (adjacent c_0_1 c_1_1)
    (adjacent c_2_0 c_2_1)
    (adjacent c_3_1 c_3_0)
    (adjacent c_1_3 c_0_3)
    (adjacent c_1_2 c_1_3)
    (adjacent c_3_3 c_2_3)
    (adjacent c_3_1 c_3_2)
    (adjacent c_4_2 c_4_1)
    (adjacent c_2_2 c_2_1)
    (adjacent c_3_3 c_3_4)
    (adjacent c_4_3 c_3_3)
    (adjacent c_1_2 c_2_2)
    (adjacent c_4_1 c_4_2)
    (adjacent c_1_0 c_2_0)
    (adjacent c_2_1 c_1_1)
    (adjacent c_3_4 c_2_4)
    (adjacent c_2_0 c_1_0)
    (adjacent c_3_2 c_3_1)
    (adjacent c_2_1 c_3_1)
    (adjacent c_0_4 c_1_4)
    (adjacent c_1_2 c_0_2)
    (adjacent c_0_2 c_0_1)
    (adjacent c_2_2 c_3_2)
    (adjacent c_3_4 c_4_4)
    (at p2 c_4_3)
    (at p1 c_4_2)
    (at p3 c_1_3)
    (at t1 c_2_3)
    (empty t1)
    (cell c_0_0)
    (cell c_0_1)
    (cell c_0_2)
    (cell c_0_3)
    (cell c_0_4)
    (cell c_1_0)
    (cell c_1_1)
    (cell c_1_2)
    (cell c_1_3)
    (cell c_1_4)
    (cell c_2_0)
    (cell c_2_1)
    (cell c_2_2)
    (cell c_2_3)
    (cell c_2_4)
    (cell c_3_0)
    (cell c_3_1)
    (cell c_3_2)
    (cell c_3_3)
    (cell c_3_4)
    (cell c_4_0)
    (cell c_4_1)
    (cell c_4_2)
    (cell c_4_3)
    (cell c_4_4)
    (package p1)
    (locatable p1)
    (package p2)
    (locatable p2)
    (package p3)
    (locatable p3)
    (truck t1)
    (locatable t1)
  )
  (:goal
    (and
      (at p1 c_2_4)
      (at p2 c_2_4)
      (at p3 c_2_4)
    )
  )
)
