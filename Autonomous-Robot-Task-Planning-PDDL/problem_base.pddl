(define (problem caso-base-limpieza)
  (:domain limpieza-aulas)

  ;; ---------------------------------------------------------
  ;; OBJETOS
  ;; ---------------------------------------------------------
  (:objects
    wall-e - robot
    pasillo aula1 aula2 aula3 residuos - ubicacion
    bolsa1 bolsa2 - bolsa
  )

  ;; ---------------------------------------------------------
  ;; ESTADO INICIAL (:init)
  ;; ---------------------------------------------------------
  (:init
    ;; Ubicación inicial del robot
    (en-robot wall-e pasillo)
    (mano-vacia wall-e)

    ;; Definición del Mapa (Adyacencias)
    ;; Asumimos que el pasillo conecta con todo (Topología Estrella)
    (conectado pasillo aula1) (conectado aula1 pasillo)
    (conectado pasillo aula2) (conectado aula2 pasillo)
    (conectado pasillo aula3) (conectado aula3 pasillo)
    (conectado pasillo residuos) (conectado residuos pasillo)
    
    ;; Definir cuál es el cuarto de residuos
    (es-cuarto-residuos residuos)

    ;; Ubicación de la suciedad (bolsas) - Ejemplo: Una en Aula 1 y otra en Aula 3
    (en-bolsa bolsa1 aula1)
    (en-bolsa bolsa2 aula3)
    
    ;; Nota: Las aulas empiezan sucias (no declaramos (aula-limpia))
  )

  ;; ---------------------------------------------------------
  ;; META (:goal)
  ;; ---------------------------------------------------------
  (:goal
    (and
      ;; 1. Todas las bolsas deben estar en el cuarto de residuos
      (en-bolsa bolsa1 residuos)
      (en-bolsa bolsa2 residuos)
      
      ;; 2. Las aulas deben haber sido limpiadas
      (aula-limpia aula1)
      (aula-limpia aula2)
      (aula-limpia aula3)
    )
  )
)