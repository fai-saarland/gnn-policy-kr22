; Automatically converted to only require STRIPS and negative preconditions

(define (domain delivery)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (cell ?x)
    (locatable ?x)
    (package ?x)
    (truck ?x)
    (at ?x1 ?x2)
    (carrying ?x1 ?x2)
    (empty ?x1)
    (adjacent ?x1 ?x2)
  )

  (:action pick-package
    :parameters (?t ?p ?x)
    :precondition (and
      (truck ?t)
      (package ?p)
      (cell ?x)
      (at ?p ?x)
      (at ?t ?x)
      (empty ?t)
    )
    :effect (and
      (not (at ?p ?x))
      (not (empty ?t))
      (carrying ?t ?p)
    )
  )

  (:action drop-package
    :parameters (?t ?p ?x)
    :precondition (and
      (truck ?t)
      (package ?p)
      (cell ?x)
      (at ?t ?x)
      (carrying ?t ?p)
    )
    :effect (and
      (empty ?t)
      (not (carrying ?t ?p))
      (at ?p ?x)
    )
  )

  (:action move
    :parameters (?t ?from ?to)
    :precondition (and
      (truck ?t)
      (cell ?from)
      (cell ?to)
      (adjacent ?from ?to)
      (at ?t ?from)
    )
    :effect (and
      (not (at ?t ?from))
      (at ?t ?to)
    )
  )
)
