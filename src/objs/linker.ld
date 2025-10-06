SECTIONS
{
  /DISCARD/ : {
      *(.init_array*)
    }

  .init_array_avx2 : ALIGN(8) {
    __start_init_array_avx2 = .;
    KEEP(*avx2/*.o(.init_array*));
    __stop_init_array_avx2 = .;
  }

  .init_array_avx512vnni : ALIGN(8) {
    __start_init_array_avx512vnni = .;
    KEEP(*avx512vnni/*.o(.init_array*));
    __stop_init_array_avx512vnni = .;
  }

  .init_array_avxvnni : ALIGN(8) {
  __start_init_array_avxvnni = .;
  KEEP(*avxvnni/*.o(.init_array*));
  __stop_init_array_avxvnni = .;
}

.init_array_bmi2 : ALIGN(8) {
  __start_init_array_bmi2 = .;
  KEEP(*bmi2/*.o(.init_array*));
  __stop_init_array_bmi2 = .;
}

.init_array_avx512icl : ALIGN(8) {
  __start_init_array_avx512icl = .;
  KEEP(*avx512icl/*.o(.init_array*));
  __stop_init_array_avx512icl = .;
}

.init_array_sse41_popcnt : ALIGN(8) {
  __start_init_array_sse41_popcnt = .;
  KEEP(*sse41-popcnt/*.o(.init_array*));
  __stop_init_array_sse41_popcnt = .;
}


  .init_array_avx512 : ALIGN(8) {
    __start_init_array_avx512 = .;
    KEEP(*avx512/*.o(.init_array*));
    __stop_init_array_avx512 = .;
  }
}
INSERT AFTER .init_array;