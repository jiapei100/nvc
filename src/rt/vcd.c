//
//  Copyright (C) 2011  Nick Gasson
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include "util.h"
#include "rt.h"
#include "tree.h"

#include <time.h>
#include <inttypes.h>
#include <string.h>

#define MAX_VAR_WIDTH  256
#define MAX_TEXT_WIDTH 512

static FILE *vcd_file = NULL;
static ident_t i_vcd_key = NULL;

static const char *vcd_key_fmt(int key)
{
   static char buf[64];

   char *p = buf;
   do {
      *p = 34 + (key % (126 - 33));
      key /= (126 - 33);
   } while (key > 0);

   return buf;
}

static int vcd_fmt_one(type_t type, char *buf, size_t max, uint64_t val)
{
   switch (type_kind(type)) {
   case T_INTEGER:
      return snprintf(buf, max, "h%"PRIx64, val);

   case T_ENUM:
      {
         const char map[] = { '0', '1', 'x', 'z' };
         if (val > sizeof(map))
            return snprintf(buf, max, "x");
         else
            return snprintf(buf, max, "%c", map[val]);
      }

   default:
      return 0; // Cannot format this in VCD
   }
}

static const char *vcd_value_fmt(tree_t decl)
{
   static char buf[MAX_TEXT_WIDTH];

   uint64_t vals[MAX_VAR_WIDTH];
   size_t w = rt_signal_value(decl, vals, MAX_VAR_WIDTH);
   (void)w;
   type_t type = tree_type(decl);
   if (type_kind(type) == T_CARRAY) {
      char *p = buf;
      const char *end = buf + MAX_TEXT_WIDTH;
      p += snprintf(p, end - p, "b");
      for (size_t i = 0; i < w; i++)
         p += vcd_fmt_one(type_base(type), p, end - p, vals[i]);
   }
   else
      vcd_fmt_one(type, buf, MAX_TEXT_WIDTH, vals[0]);

   return buf;
}

static void vcd_emit_header(void)
{
   rewind(vcd_file);

   char tmbuf[64];
   time_t t = time(NULL);
   struct tm *tm = localtime(&t);
   strftime(tmbuf, sizeof(tmbuf), "%a, %d %b %Y %T %z", tm);
   fprintf(vcd_file, "$date\n  %s\n$end\n", tmbuf);

   fprintf(vcd_file, "$version\n  "PACKAGE_STRING"\n$end\n");
}

static void vcd_event_cb(tree_t decl)
{
   uint64_t vals[MAX_VAR_WIDTH];
   size_t n = rt_signal_value(decl, vals, MAX_VAR_WIDTH);

   printf("callback! %s n=%zu\n", istr(tree_ident(decl)), n);
}

void vcd_restart(tree_t top)
{
   if (vcd_file == NULL)
      return;

   vcd_emit_header();

   fprintf(vcd_file, "$scope module %s $end\n", istr(tree_ident(top)));

   int next_key = 0;
   for (unsigned i = 0; i < tree_decls(top); i++) {
      tree_t d = tree_decl(top, i);
      if (tree_kind(d) != T_SIGNAL_DECL)
         continue;

      rt_set_event_cb(d, vcd_event_cb);

      tree_add_attr_int(d, i_vcd_key, next_key);

      type_t type = tree_type(d);
      int w = 1;
      if (type_kind(type) == T_CARRAY) {
         int64_t low, high;
         range_bounds(type_dim(type, 0), &low, &high);
         w = high - low + 1;
      }

      fprintf(vcd_file, "$var wire %d %s %s $end\n",
              w, vcd_key_fmt(next_key), istr(tree_ident(d)));

      ++next_key;
   }

   fprintf(vcd_file, "$upscope $end\n");
   fprintf(vcd_file, "$enddefinitions $end\n");

   fprintf(vcd_file, "$dumpvars\n");

   for (unsigned i = 0; i < tree_decls(top); i++) {
      tree_t d = tree_decl(top, i);
      if (tree_kind(d) != T_SIGNAL_DECL)
         continue;

      int key = tree_attr_int(d, i_vcd_key, -1);
      fprintf(vcd_file, "%s %s\n", vcd_value_fmt(d), vcd_key_fmt(key));
   }

   fprintf(vcd_file, "$end\n");
}

void vcd_init(const char *filename)
{
   i_vcd_key = ident_new("vcd_key");

   vcd_file = fopen(filename, "w");
   if (vcd_file == NULL)
      fatal_errno("failed to open VCD output %s", filename);
}
