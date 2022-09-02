; Automatically converted to only require STRIPS and negative preconditions

(define (problem reward-2x2)
  (:domain reward-strips)
  (:objects c_0_0 c_0_1 c_1_0 c_1_1)
  (:init
    (reward c_1_1)
    (at c_0_0)
    (adjacent c_1_1 c_0_1)
    (adjacent c_1_1 c_1_0)
    (adjacent c_0_0 c_0_1)
    (adjacent c_0_1 c_0_0)
    (adjacent c_1_0 c_0_0)
    (adjacent c_0_0 c_1_0)
    (adjacent c_1_0 c_1_1)
    (adjacent c_0_1 c_1_1)
    (cell c_0_0)
    (cell c_0_1)
    (cell c_1_0)
    (cell c_1_1)
  )
  (:goal
    (and
      (not (reward c_1_1))
    )
  )
)
