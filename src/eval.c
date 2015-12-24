//
//  Copyright (C) 2013-2015  Nick Gasson
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

#include "phase.h"
#include "util.h"
#include "common.h"
#include "vcode.h"

#include <assert.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <inttypes.h>

#define MAX_DIMS 4

typedef enum {
   VALUE_REAL,
   VALUE_INTEGER,
   VALUE_POINTER,
   VALUE_UARRAY,
} value_kind_t;

typedef struct value value_t;

typedef struct {
   struct {
      int64_t      left;
      int64_t      right;
      range_kind_t dir;
   } dim[MAX_DIMS];
   int      ndims;
   value_t *data;
} uarray_t;

struct value {
   value_kind_t kind;
   union {
      double   real;
      int64_t  integer;
      value_t *pointer;
      uarray_t uarray;
   };
};

typedef struct {
   value_t     *regs;
   value_t     *vars;
   int          result;
   tree_t       fcall;
   eval_flags_t flags;
   bool         failed;
} eval_state_t;

static void eval_vcode(eval_state_t *state);

static bool eval_possible(tree_t fcall, eval_flags_t flags)
{
   if (tree_attr_int(tree_ref(fcall), impure_i, 0))
      return NULL;

   const int nparams = tree_params(fcall);
   for (int i = 0; i < nparams; i++) {
      tree_t p = tree_value(tree_param(fcall, i));
      const tree_kind_t kind = tree_kind(p);
      switch (kind) {
      case T_LITERAL:
         break;

      case T_FCALL:
         if (!eval_possible(p, flags))
            return false;
         break;

      case T_REF:
         {
            const tree_kind_t dkind = tree_kind(tree_ref(p));
            if (dkind == T_UNIT_DECL || dkind == T_ENUM_LIT)
               break;
         }
         // Fall-through

      default:
         if (flags & EVAL_WARN)
            warn_at(tree_loc(p), "value of parameter %s prevents constant "
                    "folding", istr(tree_ident(tree_port(tree_ref(fcall), i))));
         return false;
      }
   }

   return true;
}

static value_t *eval_get_reg(vcode_reg_t reg, eval_state_t *state)
{
   return &(state->regs[reg]);
}

static value_t *eval_get_var(vcode_var_t var, eval_state_t *state)
{
   assert(vcode_var_context(var) == vcode_unit_depth());
   return &(state->vars[vcode_var_index(var)]);
}

static bool eval_value_eq(value_t *lhs, value_t *rhs)
{
   switch (lhs->kind) {
   case VALUE_INTEGER:
      return lhs->integer == rhs->integer;

   case VALUE_REAL:
      return lhs->real == rhs->real;

   default:
      fatal_trace("invalid value type in %s", __func__);
   }
}

static void eval_op_const(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   dst->kind    = VALUE_INTEGER;
   dst->integer = vcode_get_value(op);
}

static void eval_op_const_real(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   dst->kind = VALUE_REAL;
   dst->real = vcode_get_real(op);
}

static void eval_op_return(int op, eval_state_t *state)
{
   if (vcode_count_args(op) > 0)
      state->result = vcode_get_arg(op, 0);
}

static void eval_op_not(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *src = eval_get_reg(vcode_get_arg(op, 0), state);
   dst->kind    = VALUE_INTEGER;
   dst->integer = !(src->integer);
}

static void eval_op_add(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *lhs = eval_get_reg(vcode_get_arg(op, 0), state);
   value_t *rhs = eval_get_reg(vcode_get_arg(op, 1), state);

   switch (lhs->kind) {
   case VALUE_INTEGER:
      dst->kind    = VALUE_INTEGER;
      dst->integer = lhs->integer + rhs->integer;
      break;

   case VALUE_REAL:
      dst->kind = VALUE_REAL;
      dst->real = lhs->real + rhs->real;
      break;

   default:
      fatal_trace("invalid value type in %s", __func__);
   }
}

static void eval_op_mul(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *lhs = eval_get_reg(vcode_get_arg(op, 0), state);
   value_t *rhs = eval_get_reg(vcode_get_arg(op, 1), state);

   switch (lhs->kind) {
   case VALUE_INTEGER:
      dst->kind    = VALUE_INTEGER;
      dst->integer = lhs->integer * rhs->integer;
      break;

   case VALUE_REAL:
      dst->kind = VALUE_REAL;
      dst->real = lhs->real * rhs->real;
      break;

   default:
      fatal_trace("invalid value type in %s", __func__);
   }
}

static void eval_op_div(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *lhs = eval_get_reg(vcode_get_arg(op, 0), state);
   value_t *rhs = eval_get_reg(vcode_get_arg(op, 1), state);

   switch (lhs->kind) {
   case VALUE_INTEGER:
      if (rhs->integer == 0)
         fatal_at(tree_loc(state->fcall), "division by zero");
      else {
         dst->kind    = VALUE_INTEGER;
         dst->integer = lhs->integer / rhs->integer;
      }
      break;

   case VALUE_REAL:
      dst->kind = VALUE_REAL;
      dst->real = lhs->real / rhs->real;
      break;

   default:
      fatal_trace("invalid value type in %s", __func__);
   }
}

static void eval_op_cmp(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *lhs = eval_get_reg(vcode_get_arg(op, 0), state);
   value_t *rhs = eval_get_reg(vcode_get_arg(op, 1), state);

   dst->kind    = VALUE_INTEGER;
   dst->integer = eval_value_eq(lhs, rhs);
}

static void eval_op_cast(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *src = eval_get_reg(vcode_get_arg(op, 0), state);

   switch (vtype_kind(vcode_get_type(op))) {
   case VCODE_TYPE_INT:
   case VCODE_TYPE_OFFSET:
      dst->kind = VALUE_INTEGER;
      switch (src->kind) {
      case VALUE_INTEGER: break;
      case VALUE_REAL: dst->integer = (int64_t)src->real; break;
      default: break;
      }
      break;

   case VCODE_TYPE_REAL:
      dst->kind = VALUE_REAL;
      switch (src->kind) {
      case VALUE_INTEGER: dst->real = (double)src->integer; break;
      case VALUE_REAL: break;
      default: break;
      }
      break;

   default:
      vcode_dump();
      fatal("cannot handle destination type in cast");
   }
}

static void eval_op_neg(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *src = eval_get_reg(vcode_get_arg(op, 0), state);

   switch (src->kind) {
   case VALUE_INTEGER:
      dst->kind    = VALUE_INTEGER;
      dst->integer = -(src->integer);
      break;

   case VALUE_REAL:
      dst->kind = VALUE_REAL;
      dst->real = -(src->real);
      break;

   default:
      fatal_trace("invalid value type in %s", __func__);
   }
}

static void eval_op_fcall(int op, eval_state_t *state)
{
   vcode_state_t vcode_state;
   vcode_state_save(&vcode_state);

   ident_t func_name = vcode_get_func(op);
   vcode_unit_t vcode = vcode_find_unit(func_name);

   const int nparams = vcode_count_args(op);
   value_t *params[nparams];
   for (int i = 0; i < nparams; i++)
      params[i] = eval_get_reg(vcode_get_arg(op, i), state);

   if (vcode == NULL && (state->flags & EVAL_LOWER)) {
      ident_t unit_name = ident_runtil(vcode_get_func(op), '.');
      ident_t lib_name = ident_until(unit_name, '.');

      lib_t lib = lib_find(lib_name, false);
      if (lib != NULL) {
         tree_t unit = lib_get(lib, unit_name);
         if (unit != NULL) {
            notef("lowering %s", istr(unit_name));
            lower_unit(unit);

            if (tree_kind(unit) == T_PACKAGE) {
               tree_t body =
                  lib_get(lib, ident_prefix(unit_name, ident_new("body"), '-'));
               if (body != NULL)
                  lower_unit(body);
            }

            vcode = vcode_find_unit(func_name);
         }
      }
   }

   if (vcode == NULL) {
      if (state->flags & EVAL_WARN)
         warn_at(tree_loc(state->fcall), "function call to %s prevents "
                 "constant folding", istr(func_name));
      state->failed = true;
      vcode_state_restore(&vcode_state);
      return;
   }

   vcode_select_unit(vcode);
   vcode_select_block(0);

   value_t *regs LOCAL = xcalloc(sizeof(value_t) * vcode_count_regs());
   value_t *vars LOCAL = xcalloc(sizeof(value_t) * vcode_count_vars());

   for (int i = 0; i < nparams; i++)
      regs[i] = *params[i];

   eval_state_t new = {
      .regs   = regs,
      .vars   = vars,
      .result = -1,
      .fcall  = state->fcall,
      .failed = false,
      .flags  = state->flags | EVAL_BOUNDS
   };

   eval_vcode(&new);

   if (new.failed)
      state->failed = true;
   else {
      assert(new.result != -1);
      value_t *dst = eval_get_reg(vcode_get_result(op), state);
      *dst = regs[new.result];
   }

   vcode_state_restore(&vcode_state);
}

static void eval_op_bounds(int op, eval_state_t *state)
{
   value_t *reg = eval_get_reg(vcode_get_arg(op, 0), state);
   vcode_type_t bounds = vcode_get_type(op);

   switch (reg->kind) {
   case VALUE_INTEGER:
      {
         const int64_t low  = vtype_low(bounds);
         const int64_t high = vtype_high(bounds);
         if (low > high)
            break;
         else if (reg->integer < low || reg->integer > high)
            state->failed = true;
      }
      break;

   case VALUE_REAL:
      break;

   default:
      fatal_trace("invalid value type in %s", __func__);
   }
}

static void eval_op_const_array(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);

   const int nargs = vcode_count_args(op);

   dst->kind    = VALUE_POINTER;
   dst->pointer = xmalloc(sizeof(value_t) * nargs);   // XXX: free this
   for (int i = 0; i < nargs; i++)
      dst->pointer[i] = *eval_get_reg(vcode_get_arg(op, i), state);
}

static void eval_op_wrap(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *src = eval_get_reg(vcode_get_arg(op, 0), state);

   dst->kind = VALUE_UARRAY;
   dst->uarray.data = src;

   // XXX: fill in dims
}

static void eval_op_store(int op, eval_state_t *state)
{
   value_t *src = eval_get_reg(vcode_get_arg(op, 0), state);
   value_t *var = eval_get_var(vcode_get_address(op), state);

   *var = *src;
}

static void eval_op_unwrap(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *src = eval_get_reg(vcode_get_arg(op, 0), state);

   dst->kind    = VALUE_POINTER;
   dst->pointer = src->uarray.data;
}

static void eval_op_uarray_len(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *src = eval_get_reg(vcode_get_arg(op, 0), state);

   const int dim = vcode_get_dim(op);
   const int64_t left = src->uarray.dim[dim].left;
   const int64_t right = src->uarray.dim[dim].right;
   const range_kind_t dir = src->uarray.dim[dim].dir;

   const int64_t len = (dir == RANGE_TO ? right - left : left - right) + 1;

   dst->kind    = VALUE_INTEGER;
   dst->integer = MAX(len, 0);
}

static void eval_op_memcmp(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *lhs = eval_get_reg(vcode_get_arg(op, 0), state);
   value_t *rhs = eval_get_reg(vcode_get_arg(op, 1), state);
   value_t *len = eval_get_reg(vcode_get_arg(op, 2), state);

   dst->kind    = VALUE_INTEGER;
   dst->integer = 1;

   for (int i = 0; i < len->integer; i++) {
      if (!eval_value_eq(&(lhs->pointer[i]), &(rhs->pointer[i]))) {
         dst->integer = 0;
         return;
      }
   }
}

static void eval_vcode(eval_state_t *state)
{
   const int nops = vcode_count_ops();
   for (int i = 0; i < nops && !(state->failed); i++) {
      switch (vcode_get_op(i)) {
      case VCODE_OP_COMMENT:
         break;

      case VCODE_OP_CONST:
         eval_op_const(i, state);
         break;

      case VCODE_OP_CONST_REAL:
         eval_op_const_real(i, state);
         break;

      case VCODE_OP_RETURN:
         eval_op_return(i, state);
         return;

      case VCODE_OP_NOT:
         eval_op_not(i, state);
         break;

      case VCODE_OP_ADD:
         eval_op_add(i, state);
         break;

      case VCODE_OP_MUL:
         eval_op_mul(i, state);
         break;

      case VCODE_OP_DIV:
         eval_op_div(i, state);
         break;

      case VCODE_OP_CMP:
         eval_op_cmp(i, state);
         break;

      case VCODE_OP_CAST:
         eval_op_cast(i, state);
         break;

      case VCODE_OP_NEG:
         eval_op_neg(i, state);
         break;

      case VCODE_OP_FCALL:
         if (state->flags & EVAL_FCALL)
            eval_op_fcall(i, state);
         else
            state->failed = true;
         break;

      case VCODE_OP_BOUNDS:
         eval_op_bounds(i, state);
         break;

      case VCODE_OP_CONST_ARRAY:
         eval_op_const_array(i, state);
         break;

      case VCODE_OP_WRAP:
         eval_op_wrap(i, state);
         break;

      case VCODE_OP_STORE:
         eval_op_store(i, state);
         break;

      case VCODE_OP_UNWRAP:
         eval_op_unwrap(i, state);
         break;

      case VCODE_OP_UARRAY_LEN:
         eval_op_uarray_len(i, state);
         break;

      case VCODE_OP_MEMCMP:
         eval_op_memcmp(i, state);
         break;

      default:
         vcode_dump();
         fatal("cannot evaluate vcode op %s", vcode_op_string(vcode_get_op(i)));
      }
   }
}

tree_t eval(tree_t fcall, eval_flags_t flags)
{
   assert(tree_kind(fcall) == T_FCALL);

   if (!type_is_scalar(tree_type(fcall)))
      return false;

   if (!eval_possible(fcall, flags))
      return fcall;

   vcode_unit_t thunk = lower_thunk(fcall);
   if (thunk == NULL)
      return fcall;

   vcode_select_unit(thunk);
   vcode_select_block(0);

   value_t *regs LOCAL = xcalloc(sizeof(value_t) * vcode_count_regs());

   eval_state_t state = {
      .regs   = regs,
      .result = -1,
      .fcall  = fcall,
      .failed = false,
      .flags  = flags
   };

   eval_vcode(&state);

   if (state.failed)
      return fcall;

   assert(state.result != -1);
   value_t result = regs[state.result];
   #if 0
   fmt_loc(stdout, tree_loc(fcall));
   if (result.kind == VALUE_INTEGER)
      printf("result=%"PRIi64"\n", result.integer);
   else
      printf("result=%lf\n", result.real);
   #endif

   switch (result.kind) {
   case VALUE_INTEGER:
      return get_int_lit(fcall, result.integer);
   case VALUE_REAL:
      return get_real_lit(fcall, result.real);
   default:
      fatal_trace("eval result is not scalar");
   }
}
