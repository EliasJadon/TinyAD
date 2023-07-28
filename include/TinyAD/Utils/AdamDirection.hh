#pragma once

#include <Eigen/Core>
#include <TinyAD/Utils/Out.hh>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Utils/LinearSolver.hh>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>

namespace TinyAD_ext
{
    /**
     * Compute update vector d such that x + d performs a ADAM step
     * Input:
     *      _g: gradient
     */
    template <typename PassiveT>
    Eigen::VectorX<PassiveT> adam_direction(
        const Eigen::VectorX<PassiveT>& _g,
        const double beta1 = 0.9,
        const double beta2 = 0.999,
        const double epsilon = 1e-8)
    {
        static Eigen::VectorX<PassiveT> v = Eigen::VectorX<PassiveT>::Zero(_g.size());
        static Eigen::VectorX<PassiveT> s = Eigen::VectorX<PassiveT>::Zero(_g.size());
        Eigen::VectorX<PassiveT> d = Eigen::VectorX<PassiveT>::Zero(_g.size());

        for (int i = 0; i < _g.size(); i++) {
            v[i] = beta1 * v[i] + (1 - beta1) * _g[i];
            s[i] = beta2 * s[i] + (1 - beta2) * pow(_g[i], 2);
            d[i] = -v[i] / (sqrt(s[i]) + epsilon);
        }
        TINYAD_ASSERT_FINITE_MAT(d);
        return d;
    }
}
