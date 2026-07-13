/* System call checks
   Copyright (C) 2025  Peter Behroozi

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   If so, it should be in a file called "LICENSE".
*/
#ifndef CHECK_SYSCALLS_H
#define CHECK_SYSCALLS_H
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>

void system_error(char *errmsg);
FILE *check_fopen(char *filename, char *mode);
FILE *check_fdopen(int fd, char *mode);
FILE *check_create_fopen(char *filename, char *mode);
FILE *check_popen(char *command, char *mode);
void *check_realloc(void *ptr, size_t size, char *reason);
void *check_realloc_aligned(void *ptr, size_t prev_size, size_t size, char *reason);
void check_fstat(int fd, struct stat *ts, char *filename);
size_t check_fread(void *ptr, size_t size, size_t nitems, FILE *stream);
void check_limited_funread(void *ptr, size_t size, size_t nitems);
size_t check_fwrite(void *ptr, size_t size, size_t nitems, FILE *stream);
void check_slurp_file(char *fn, void **data, int64_t *length);
void check_fseeko(FILE *stream, off_t offset, int whence);
void check_lseek(int fd, off_t offset, int whence);
char *check_fgets(char *ptr, size_t size, FILE *stream);
char *check_fgets_and_chomp(char *ptr, size_t size, FILE *stream);
FILE *check_rw_socket(char *command, pid_t *pid);
void rw_socket_close(FILE *res, pid_t pid);
void *check_mmap_file(char *filename, char mode, int64_t *length);
void *check_mmap_memory(int64_t length);
pid_t check_waitpid(pid_t pid);
void check_fskip(FILE *stream, off_t offset, char *buffer, size_t buf_size);
void check_mtrim(void);
pid_t check_fork(void);
void check_rename(const char *old, const char *new);
void check_mkdir(char *dirname, mode_t mode);
char *check_strdup(const char *str);
void check_pipe(int filedes[2]);
#define check_free(x) { if (x) { free(x); } (x) = NULL; }

  
#define check_fprintf(file, ...) { if (fprintf(file, __VA_ARGS__) <= 0)	{  \
      fprintf(stderr, "[Error] Failed printf to fileno %d!\n", fileno(file)); \
      perror("[Error] Reason"); \
      exit(1); \
    }}


#define check_log(file, ...) { if (file) { check_fprintf(file, __VA_ARGS__); fflush(file); } }

#define stringify(x) #x
#define to_string(x) stringify(x)
#define check_realloc_s(x,y,z) { (x) = check_realloc((x),((int64_t)(y))*((int64_t)(z)), "Reallocating " #x " at " __FILE__ ":" to_string(__LINE__)); }
#define check_calloc_s(x,y,z) { check_realloc_s(x,y,z); memset(x, 0, ((int64_t)(y))*((int64_t)(z))); }
#define check_realloc_aligned_s(x,s,y,z) { (x) = check_realloc_aligned((x),((int64_t)(y))*((int64_t)(s)),((int64_t)(z))*((int64_t)(s)), "Reallocating " #x " at " __FILE__ ":" to_string(__LINE__)); }

#define check_realloc_var(x,size,cur,new) { if (cur < new) { cur = new; check_realloc_s(x,size,new); } }
#define check_realloc_every(x,size,cur,num) { if (!((cur)%(num))) { check_realloc_s(x,size,(cur)+(num)); } }
#define check_realloc_smart(x,size,cur,new) { if ((cur) < (new)) { cur = (new)*1.05 + 1000; check_realloc_s(x,size,cur); } }
#define check_realloc_bold(x,size,cur,new) { if ((cur) < (new)) { cur = (new)*1.3 + 1000; check_realloc_s(x,size,cur); } }
#define check_realloc_bold_aligned(x,size,cur,new) { if ((cur) < (new)) { cur = (new)*1.3 + 1000; check_realloc_aligned_s(x,size,new,cur); } }
#define check_realloc_aggressive(x,size,cur,new) { if ((cur) < (new)) { cur = (new)*2.0 + 1000; check_realloc_s(x,size,cur); } }

extern FILE *syscall_logfile;

#endif /* CHECK_SYSCALLS_H */
