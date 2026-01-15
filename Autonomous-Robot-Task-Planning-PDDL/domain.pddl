(define (domain limpieza-aulas)
  (:requirements :strips :typing)

  ;; ---------------------------------------------------------
  ;; DEFINICIÓN DE TIPOS
  ;; Ayuda a evitar errores lógicos (ej: limpiar una bolsa)
  ;; ---------------------------------------------------------
  (:types
    ubicacion      ; Lugares (aulas, pasillo, residuos)
    bolsa          ; Las bolsas de basura
    robot          ; El agente
  )

  ;; ---------------------------------------------------------
  ;; PREDICADOS (Variables de estado)
  ;; ---------------------------------------------------------
  (:predicates
    (en-robot ?r - robot ?u - ubicacion)       ; Dónde está el robot
    (en-bolsa ?b - bolsa ?u - ubicacion)       ; Dónde está una bolsa
    (conectado ?u1 - ubicacion ?u2 - ubicacion); Mapa de navegación
    
    (mano-vacia ?r - robot)                    ; Si el robot no carga nada
    (sosteniendo ?r - robot ?b - bolsa)        ; Qué bolsa lleva el robot
    
    (es-cuarto-residuos ?u - ubicacion)        ; Marca especial para el cuarto de basura
    (aula-limpia ?u - ubicacion)               ; Estado de limpieza del aula
  )

  ;; ---------------------------------------------------------
  ;; ACCIONES
  ;; ---------------------------------------------------------

  ;; 1. MOVERSE: Ir de un lugar a otro adyacente
  (:action mover
    :parameters (?r - robot ?origen - ubicacion ?destino - ubicacion)
    :precondition (and 
        (en-robot ?r ?origen)
        (conectado ?origen ?destino)
    )
    :effect (and 
        (not (en-robot ?r ?origen))
        (en-robot ?r ?destino)
    )
  )

  ;; 2. RECOGER BOLSA: Solo si el robot y la bolsa están en el mismo sitio
  (:action recoger
    :parameters (?r - robot ?b - bolsa ?u - ubicacion)
    :precondition (and 
        (en-robot ?r ?u)
        (en-bolsa ?b ?u)
        (mano-vacia ?r)
    )
    :effect (and 
        (not (en-bolsa ?b ?u))
        (not (mano-vacia ?r))
        (sosteniendo ?r ?b)
    )
  )

  ;; 3. DEPOSITAR BOLSA: Restricción -> SOLO en cuarto de residuos
  (:action depositar
    :parameters (?r - robot ?b - bolsa ?u - ubicacion)
    :precondition (and 
        (en-robot ?r ?u)
        (sosteniendo ?r ?b)
        (es-cuarto-residuos ?u)  ; <--- REGLA DE ORO DEL DOCUMENTO
    )
    :effect (and 
        (not (sosteniendo ?r ?b))
        (mano-vacia ?r)
        (en-bolsa ?b ?u)
    )
  )

  ;; 4. LIMPIAR AULA: El robot limpia el piso
  (:action limpiar
    :parameters (?r - robot ?u - ubicacion)
    :precondition (and 
        (en-robot ?r ?u)
        ;; Podríamos agregar (not (es-cuarto-residuos ?u)) si no se limpia la basura
    )
    :effect (and 
        (aula-limpia ?u)
    )
  )
)