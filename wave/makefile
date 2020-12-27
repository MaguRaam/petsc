-include ../petscdir.mk
CFLAGS		=	
CPPFLAGS	=	-Wall
LIBFILES	=
TARGET		=	wave
SRC			=	$(wildcard *.cc)
OBJ			=	$(SRC:.cc=.o)
CLEANFILES	=	$(TARGET)
LOCDIR		=	$(CURDIR)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

all: $(TARGET)

$(TARGET): $(OBJ)
		-${CLINKER} -o $(TARGET) $(OBJ) ${PETSC_LIB}
		${RM} $(OBJ)