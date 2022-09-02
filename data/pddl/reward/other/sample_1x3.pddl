; Automatically converted to only require STRIPS and negative preconditions

(define (problem reward-4x4)
  (:domain reward-strips)
  (:objects c_0_0 c_0_1 c_0_2)
  (:init
    (reward c_0_2)
    (at c_0_0)
    (adjacent c_0_0 c_0_1)
    (adjacent c_0_1 c_0_0)
    (adjacent c_0_2 c_0_1)
    (adjacent c_0_1 c_0_2)
    (cell c_0_0)
    (cell c_0_1)
    (cell c_0_2)
  )
  (:goal
    (and
      (not (reward c_0_1))
      (not (reward c_0_2))
    )
  )
)
