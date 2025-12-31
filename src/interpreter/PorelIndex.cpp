/*
 * Souffle - A Datalog Compiler
 * Copyright (c) 2019, The Souffle Developers. All rights reserved.
 * Licensed under the Universal Permissive License v 1.0 as shown at:
 * - https://opensource.org/licenses/UPL
 * - <souffle root>/licenses/SOUFFLE-UPL.txt
 */

/************************************************************************
 *
 * @file PorelIndex.cpp
 *
 * Interpreter index with generic interface.
 *
 ***********************************************************************/

#include "interpreter/Relation.h"
#include "ram/Relation.h"
#include "ram/analysis/Index.h"

namespace souffle::interpreter {

Own<RelationWrapper> createPorelRelation(
        const ram::Relation& id, const ram::analysis::IndexCluster& indexSelection) {
    assert(id.getArity() == 2 && "Transitive relation must have arity size 2.");
    assert(id.getAuxiliaryArity() == 0 && "Transitive relation must have auxiliary arity size 0.");
    return mk<PorelRelation>(id.getName(), indexSelection);
}

}  // namespace souffle::interpreter
