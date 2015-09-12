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

typedef enum {
   VALUE_REAL,
   VALUE_INTEGER,
} value_kind_t;

typedef struct {
   value_kind_t kind;
   union {
      double  real;
      int64_t integer;
   };
} value_t;

typedef struct {
   value_t *regs;
   int      result;
   tree_t   fcall;
} eval_state_t;

static bool eval_possible(tree_t fcall)
{
   type_t result = tree_type(fcall);
   if (!type_is_scalar(result))
      return false;

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
         if (!eval_possible(p))
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
         return false;
      }
   }

   return true;
}

static value_t *eval_get_reg(vcode_reg_t reg, eval_state_t *state)
{
   return &(state->regs[reg]);
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
   }
}

static void eval_op_cmp(int op, eval_state_t *state)
{
   value_t *dst = eval_get_reg(vcode_get_result(op), state);
   value_t *lhs = eval_get_reg(vcode_get_arg(op, 0), state);
   value_t *rhs = eval_get_reg(vcode_get_arg(op, 1), state);

   dst->kind = VALUE_INTEGER;

   switch (lhs->kind) {
   case VALUE_INTEGER:
      dst->integer = lhs->integer == rhs->integer;
      break;

   case VALUE_REAL:
      dst->real = lhs->real == rhs->real;
      break;
   }
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
      }
      break;

   case VCODE_TYPE_REAL:
      dst->kind = VALUE_REAL;
      switch (src->kind) {
      case VALUE_INTEGER: dst->real = (double)src->integer; break;
      case VALUE_REAL: break;
      }
      break;

   default:
      vcode_dump();
      fatal("cannot handle destination type in cast");
   }
}

static void eval_vcode(eval_state_t *state)
{
   const int nops = vcode_count_ops();
   for (int i = 0; i < nops; i++) {
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

      default:
         vcode_dump();
         fatal("cannot evaluate vcode op %s", vcode_op_string(vcode_get_op(i)));
      }
   }
}

static value_t eval_thunk(tree_t fcall, vcode_unit_t thunk)
{
   vcode_select_unit(thunk);
   vcode_select_block(0);

   eval_state_t state = {
      .regs   = xcalloc(sizeof(value_t) * vcode_count_regs()),
      .result = -1,
      .fcall  = fcall
   };

   eval_vcode(&state);

   assert(state.result != -1);
   value_t result = state.regs[state.result];

   free(state.regs);
   return result;
}

tree_t eval(tree_t fcall)
{
   assert(tree_kind(fcall) == T_FCALL);

   if (!eval_possible(fcall))
      return fcall;

   vcode_unit_t thunk = lower_thunk(fcall);
   if (thunk == NULL)
      return fcall;

   value_t result = eval_thunk(fcall, thunk);
   if (result.kind == VALUE_INTEGER)
      printf("result=%"PRIi64"\n", result.integer);
   else
      printf("result=%lf\n", result.real);

   switch (result.kind) {
   case VALUE_INTEGER:
      return get_int_lit(fcall, result.integer);
   case VALUE_REAL:
      return get_real_lit(fcall, result.real);
   }
}
