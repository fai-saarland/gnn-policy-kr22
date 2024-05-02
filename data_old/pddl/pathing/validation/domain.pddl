; Automatically converted to only require STRIPS and negative preconditions

(define (domain grid-path)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (connected ?x ?y)
    (at-robot ?x)
  )

  (:action move
    :parameters (?curpos ?nextpos)
    :precondition (and
      (at-robot ?curpos)
      (connected ?curpos ?nextpos)
    )
    :effect (and
      (at-robot ?nextpos)
      (not (at-robot ?curpos))
    )
  )
)
