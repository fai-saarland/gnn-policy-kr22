; Automatically converted to only require STRIPS and negative preconditions

(define (domain reward-strips)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (cell ?x)
    (at ?x1)
    (reward ?x1)
    (unblocked ?x1)
    (picked ?x1)
    (adjacent ?x1 ?x2)
  )

  (:action move
    :parameters (?from ?to)
    :precondition (and
      (cell ?from)
      (cell ?to)
      (adjacent ?from ?to)
      (at ?from)
      (unblocked ?to)
    )
    :effect (and
      (not (at ?from))
      (at ?to)
    )
  )

  (:action pick-reward
    :parameters (?x)
    :precondition (and
      (cell ?x)
      (at ?x)
      (reward ?x)
    )
    :effect (and
      (not (reward ?x))
      (picked ?x)
    )
  )
)
