#pragma once

#include <deal.II/base/numbers.h>

#include <algorithm>
#include <ammber/core/quadratic_pde.h>
#include <array>
#include <cmath>
#include <limits>
#include <prismspf/core/pde_operator_base.h>
#include <prismspf/utilities/logger.h>
#include <vector>

PRISMS_PF_BEGIN_NAMESPACE

/**
 * Ternary eutectic LIQUID -> alpha-Al (FCC_A1) + Ag2Al (HCP_A3) + Al2Cu (THETA).
 *
 * Initial condition: a eutectic seed of height `seed_height` along the bottom of
 * the growth axis (y in 2D, z in 3D), undercooled liquid above it. The seed
 * pattern is chosen in set_initial_condition, which calls one of
 *
 * - lamellar_seed: `num_periods` ABAC periods across x,
 *
 *       | Al2Cu/2 | alpha | Ag2Al | alpha | Al2Cu/2 |
 *
 *   with lamella widths set by the lever rule, i.e. the solid fractions whose
 *   average composition (from each phase's c0) equals the liquid's c0. Each
 *   period is mirror-symmetric about its ends, so the seed is consistent with
 *   both periodic and zero-flux (natural) boundary conditions in x. In 3D the
 *   lamellae are plates normal to x.
 * - hexagonal_seed (3D only): the three solids on the three sublattices of a
 *   hexagonal pattern in the x-y plane (see hexagons), `num_periods` pattern
 *   periods across x. Each solid takes a third of the area, not its lever-rule
 *   fraction.
 */
template <unsigned int dim, unsigned int degree, typename number, unsigned int num_comps>
class CustomPDE : public QuadraticGrandPotentialPDE<dim, degree, number, num_comps>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;
  using QuadraticGrandPotentialPDE<dim, degree, number, num_comps>::sys;
  using QuadraticGrandPotentialPDE<dim, degree, number, num_comps>::submit_ic_from_fields;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  static_assert(dim >= 2, "The seed needs a growth direction");
  static_assert(num_comps == 2, "The lever rule below is written for a ternary system");

  // Order parameter indices, in the order of "order_parameters" in the system JSON:
  // ["LIQUID", "FCC_A1_0", "HCP_A3", "THETA"]
  static constexpr unsigned int liquid = 0;
  static constexpr unsigned int alpha  = 1; // alpha-Al, FCC_A1
  static constexpr unsigned int ag2al  = 2; // Ag2Al, HCP_A3
  static constexpr unsigned int al2cu  = 3; // Al2Cu, THETA

  /// Growth axis: the seed sits at its lower bound (y in 2D, z in 3D)
  static constexpr unsigned int growth = dim - 1;

  /// One lamella of a period, in fractions of the period [start, end]
  struct Lamella
  {
    unsigned int op;
    double       start;
    double       end;
  };

  double                seed_height;
  bool                  scale_with_domain = false;
  int                   num_periods;
  double                period;
  std::array<double, 3> solid_fraction {}; // alpha, Ag2Al, Al2Cu
  std::vector<Lamella>  lamellae;

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim>   &_user_inputs,
            PhaseFieldTools<dim>             &_pf_tools,
            const QuadraticSystem<num_comps> &_sys)
    : QuadraticGrandPotentialPDE<dim, degree, number, num_comps>(_user_inputs,
                                                                 _pf_tools,
                                                                 _sys)
    , seed_height(get_user_inputs().user_constants.get_double("seed_height"))
    , scale_with_domain(get_user_inputs().user_constants.get_bool("scale_with_domain"))
    , num_periods(get_user_inputs().user_constants.get_int("num_periods"))
    , period(get_user_inputs().user_constants.get_double("period"))
  {
    AssertThrow(sys.num_ops() == 4,
                dealii::ExcMessage("Expected 4 order parameters: LIQUID, FCC_A1_0, "
                                   "HCP_A3, THETA"));
    AssertThrow(num_periods > 0, dealii::ExcMessage("num_periods must be positive"));
    compute_solid_fractions();

    // | Al2Cu/2 | alpha | Ag2Al | alpha | Al2Cu/2 |: the Al2Cu lamella is one lamella
    // centred on the period boundary (the periodic wrap in set_initial_condition
    // joins its halves), alpha is split into two lamellae
    const double f_a = solid_fraction[0], f_z = solid_fraction[1],
                 f_t = solid_fraction[2];
    double pos       = -0.5 * f_t;
    for (const auto &[op, width] : std::vector<std::pair<unsigned int, double>> {
           {al2cu, f_t      },
           {alpha, 0.5 * f_a},
           {ag2al, f_z      },
           {alpha, 0.5 * f_a}
    })
      {
        lamellae.push_back({op, pos, pos + width});
        pos += width;
      }

    Logger::instance() << "Eutectic seed: solid fractions alpha-Al " << f_a << ", Ag2Al "
                       << f_z << ", Al2Cu " << f_t << "; " << num_periods
                       << " ABAC period(s)\n";

    if constexpr (dim == 3)
      {
        // hexagonal_seed repeats every 2 scale / sqrt(3) in y
        const auto  &mesh     = get_user_inputs().spatial_discretization.rectangular_mesh;
        const double y_period = mesh.size[0] / num_periods / std::sqrt(3.0);
        const double n_y      = mesh.size[1] / y_period;
        if (std::abs(n_y - std::round(n_y)) > 1e-6 * n_y)
          Logger::instance() << "Note: y size is " << n_y
                             << " hexagonal_seed periods; use a whole number for "
                                "periodic boundaries in y\n";
      }
  }

  /**
   * @brief User-implemented class for the setting initial conditions.
   */
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    // Pick the seed pattern here
    const auto  &mesh    = get_user_inputs().spatial_discretization.rectangular_mesh;
    const double period_ = scale_with_domain ? mesh.size[0] / num_periods : period;
    // const std::vector<number> eta0 = lamellar_seed(point, period_);
    const std::vector<number> eta0 = hexagonal_seed(point, period_);

    // Submit eta0 to utility that sets ICs
    submit_ic_from_fields(eta0, index, scalar_value);
  }

  /**
   * @brief Signed distance to the seed's top surface, positive inside the seed.
   */
  double
  seed_front(const dealii::Point<dim> &point) const
  {
    const auto &mesh = get_user_inputs().spatial_discretization.rectangular_mesh;
    return mesh.lower_bound[growth] + seed_height - point[growth];
  }

  /**
   * @brief Order parameters of the lamellar ABAC seed (see the class comment).
   */
  std::vector<number>
  lamellar_seed(const dealii::Point<dim> &point, double scale) const
  {
    const auto  &mesh  = get_user_inputs().spatial_discretization.rectangular_mesh;
    const double front = seed_front(point);

    // Position within the period, as a fraction in [0, 1)
    double s = (point[0] - mesh.lower_bound[0]) / scale;
    s -= std::floor(s);

    std::vector<number> eta0(sys.num_ops(), 0.0);
    eta0[liquid] = sys.interface(-front);
    for (const Lamella &lamella : lamellae)
      {
        // Signed distance into the lamella (periodic in x), intersected with the seed
        double inside = -std::numeric_limits<double>::max();
        for (const double shift : {-1.0, 0.0, 1.0})
          {
            const double t = s + shift;
            inside =
              std::max(inside, std::min(t - lamella.start, lamella.end - t) * scale);
          }
        eta0[lamella.op] =
          std::max(eta0[lamella.op], number(sys.interface(std::min(inside, front))));
      }
    return eta0;
  }

  /**
   * @brief Order parameters of the hexagonal seed (3D only, see the class comment).
   *
   * The pattern repeats every 2 scale in x and every 2 scale / sqrt(3) in y, with
   * scale = (x size) / (2 num_periods). For periodic boundaries in y the y size
   * must be a whole number of y periods (a warning is logged if it is not).
   */
  std::vector<number>
  hexagonal_seed(const dealii::Point<dim> &point, double scale) const
  {
    AssertThrow(dim == 3, dealii::ExcMessage("hexagonal_seed needs a 3D domain"));
    // Solid phase on each sublattice of the pattern, hexagons(...)[i]
    constexpr std::array<unsigned int, 3> hex_phases = {alpha, ag2al, al2cu};

    const auto  &mesh  = get_user_inputs().spatial_discretization.rectangular_mesh;
    const double front = seed_front(point);
    const std::array<double, 3> w = hexagons(point[0], point[1], scale);

    std::vector<number> eta0(sys.num_ops(), 0.0);
    eta0[liquid] = sys.interface(-front);
    for (unsigned int i = 0; i < 3; i++)
      {
        eta0[hex_phases[i]] = sys.interface(std::min(w[i], front));
      }
    return eta0;
  }

  /**
   * @brief User-implemented function for setting Dirichlet boundary conditions. Default
   * behavior is to call initial conditions.
   */
  /* virtual void
  set_dirichlet([[maybe_unused]] const unsigned int       &index,
                [[maybe_unused]] const unsigned int       &boundary_id,
                [[maybe_unused]] const unsigned int       &component,
                [[maybe_unused]] const dealii::Point<dim> &point,
                [[maybe_unused]] const SimulationTimer    &sim_timer,
                [[maybe_unused]] number                   &scalar_value,
                [[maybe_unused]] number                   &vector_component_value) const
  override; */

  /**
   * @brief arr[i] goes into interface function
   */
  static std::array<double, 3>
  hexagons(double x, double y, double scale, double offset = 1.0, double m = 0.0)
  {
    constexpr std::array<double, 3> a = {
      {0.0, 1.0, 2.0}
    };
    constexpr double                           tau = 2.0 * dealii::numbers::PI;
    const double                               s   = 2.0 / (3.0 * std::sqrt(3.0));
    const std::array<std::array<double, 2>, 3> b   = {
      {{{std::cos(tau * a[0] / 3.0), std::sin(tau * a[0] / 3.0)}},
       {{std::cos(tau * a[1] / 3.0), std::sin(tau * a[1] / 3.0)}},
       {{std::cos(tau * a[2] / 3.0), std::sin(tau * a[2] / 3.0)}}}
    };
    const std::array<std::array<double, 2>, 3> d = {
      {{{std::cos(tau * a[0] / 3.0 + tau / 12.0),
         std::sin(tau * a[0] / 3.0 + tau / 12.0)}},
       {{std::cos(tau * a[1] / 3.0 + tau / 12.0),
         std::sin(tau * a[1] / 3.0 + tau / 12.0)}},
       {{std::cos(tau * a[2] / 3.0 + tau / 12.0),
         std::sin(tau * a[2] / 3.0 + tau / 12.0)}}}
    };

    std::array<double, 3> w = {
      {0.0, 0.0, 0.0}
    };
    for (uint i = 0; i < 3; i++)
      {
        for (uint j = 0; j < 3; j++)
          {
            w[i] += std::cos(tau * ((x / scale + offset * s * d[i][0]) * b[j][0] +
                                    (y / scale + offset * s * d[i][1]) * b[j][1] + m));
          }
        w[i] *= scale / tau;
      }
    return w;
  }

private:
  /**
   * Lever rule: solid fractions f with sum_i f_i c0_i = c0_liquid and sum_i f_i = 1,
   * solved by Cramer's rule. Negative fractions (liquid c0 outside the solids'
   * triangle) are clipped to zero and the rest renormalized.
   */
  void
  compute_solid_fractions()
  {
    const auto &c_l = sys.phases[sys.order_params[liquid]].c0;
    const auto &c_a = sys.phases[sys.order_params[alpha]].c0;
    const auto &c_z = sys.phases[sys.order_params[ag2al]].c0;
    const auto &c_t = sys.phases[sys.order_params[al2cu]].c0;
    const auto  det = [](double a, double b, double c, double d)
      {
        return a * d - b * c;
      };
    // Barycentric coordinates of c_l in the triangle (c_a, c_z, c_t)
    const double area =
      det(c_z[0] - c_a[0], c_t[0] - c_a[0], c_z[1] - c_a[1], c_t[1] - c_a[1]);
    AssertThrow(std::abs(area) > 1e-12,
                dealii::ExcMessage("The solid c0 compositions are collinear"));
    const double f_z =
      det(c_l[0] - c_a[0], c_t[0] - c_a[0], c_l[1] - c_a[1], c_t[1] - c_a[1]) / area;
    const double f_t =
      det(c_z[0] - c_a[0], c_l[0] - c_a[0], c_z[1] - c_a[1], c_l[1] - c_a[1]) / area;
    solid_fraction = {1.0 - f_z - f_t, f_z, f_t};

    if (*std::min_element(solid_fraction.begin(), solid_fraction.end()) < 0.0)
      {
        Logger::instance() << "Warning: liquid c0 is outside the triangle of solid c0; "
                              "clipping negative lever-rule fractions\n";
        double total = 0.0;
        for (double &f : solid_fraction)
          total += (f = std::max(f, 0.0));
        for (double &f : solid_fraction)
          f /= total;
      }
  }
};

PRISMS_PF_END_NAMESPACE
