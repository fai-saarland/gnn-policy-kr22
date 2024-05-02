; Automatically converted to only require STRIPS and negative preconditions

(define (domain parking)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (car ?x)
    (curb ?x)
    (at-curb ?car)
    (at-curb-num ?car ?curb)
    (behind-car ?car ?front-car)
    (car-clear ?car)
    (curb-clear ?curb)
  )

  (:action move-curb-to-curb
    :parameters (?car ?curbsrc ?curbdest)
    :precondition (and
      (car ?car)
      (curb ?curbsrc)
      (curb ?curbdest)
      (car-clear ?car)
      (curb-clear ?curbdest)
      (at-curb-num ?car ?curbsrc)
    )
    :effect (and
      (not (curb-clear ?curbdest))
      (curb-clear ?curbsrc)
      (at-curb-num ?car ?curbdest)
      (not (at-curb-num ?car ?curbsrc))
    )
  )

  (:action move-curb-to-car
    :parameters (?car ?curbsrc ?cardest)
    :precondition (and
      (car ?car)
      (curb ?curbsrc)
      (car ?cardest)
      (car-clear ?car)
      (car-clear ?cardest)
      (at-curb-num ?car ?curbsrc)
      (at-curb ?cardest)
    )
    :effect (and
      (not (car-clear ?cardest))
      (curb-clear ?curbsrc)
      (behind-car ?car ?cardest)
      (not (at-curb-num ?car ?curbsrc))
      (not (at-curb ?car))
    )
  )

  (:action move-car-to-curb
    :parameters (?car ?carsrc ?curbdest)
    :precondition (and
      (car ?car)
      (car ?carsrc)
      (curb ?curbdest)
      (car-clear ?car)
      (curb-clear ?curbdest)
      (behind-car ?car ?carsrc)
    )
    :effect (and
      (not (curb-clear ?curbdest))
      (car-clear ?carsrc)
      (at-curb-num ?car ?curbdest)
      (not (behind-car ?car ?carsrc))
      (at-curb ?car)
    )
  )

  (:action move-car-to-car
    :parameters (?car ?carsrc ?cardest)
    :precondition (and
      (car ?car)
      (car ?carsrc)
      (car ?cardest)
      (car-clear ?car)
      (car-clear ?cardest)
      (behind-car ?car ?carsrc)
      (at-curb ?cardest)
    )
    :effect (and
      (not (car-clear ?cardest))
      (car-clear ?carsrc)
      (behind-car ?car ?cardest)
      (not (behind-car ?car ?carsrc))
    )
  )
)
