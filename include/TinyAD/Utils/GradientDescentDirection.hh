#pragma once

#include <Eigen/Core>
#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Utils/LinearSolver.hh>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>

namespace TinyAD_ext
{
    /**
     * Compute update vector d such that x + d performs a Gradient-Descent step
     * Input:
     *      _g: gradient
     */
    template <typename PassiveT>
    Eigen::VectorX<PassiveT> gradient_descent_direction(const Eigen::VectorX<PassiveT>& _g)
    {
        const Eigen::VectorX<PassiveT> d = -_g;
        TINYAD_ASSERT_FINITE_MAT(d);
        return d;
    }
}
