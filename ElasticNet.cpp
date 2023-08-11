#include <Rcpp.h>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// Fonction pour effectuer la régression ElasticNet étape par étape
// [[Rcpp::export]]
List regressionElasticNet(const arma::mat& X, const arma::vec& y, double lambda, double alpha) {
  int n = X.n_rows; // Nombre d'échantillons
  int p = X.n_cols; // Nombre de variables indépendantes

  // Étape 1 : Standardisation des données
  arma::mat X_std = arma::normalise(X);

  // Étape 2 : Initialisation des coefficients à zéro
  arma::vec coefficients(p, arma::fill::zeros);

  // Étape 3 : Boucle de mise à jour des coefficients
  int max_iterations = 1000; // Nombre maximal d'itérations
  double tolerance = 0.0001; // Tolérance pour la convergence
  for (int iteration = 0; iteration < max_iterations; iteration++) {
    double max_change = 0.0; // Maximum du changement des coefficients

    // Mise à jour des coefficients un par un
    for (int j = 0; j < p; j++) {
      double old_coefficient = coefficients(j);

      // Calcul du résidu partiel
      arma::vec partial_residual = y - X_std * coefficients + X_std.col(j) * old_coefficient;

      // Calcul du coefficient partiel
      double partial_coefficient = arma::dot(X_std.col(j), partial_residual) / n;

      // Application de la pénalité ElasticNet
      double shrinkage = lambda * alpha / n;
      double l1_penalty = shrinkage;
      double l2_penalty = shrinkage * (1 - alpha);

      if (partial_coefficient > l1_penalty) {
        coefficients(j) = (partial_coefficient - l1_penalty) / (1 + l2_penalty);
      } else if (partial_coefficient < -l1_penalty) {
        coefficients(j) = (partial_coefficient + l1_penalty) / (1 + l2_penalty);
      } else {
        coefficients(j) = 0.0;
      }

      // Calcul du changement du coefficient
      double change = std::abs(coefficients(j) - old_coefficient);
      if (change > max_change) {
        max_change = change;
      }
    }

    // Vérification de la convergence
    if (max_change < tolerance) {
      break;
    }
  }

  // Étape 4 : Prédiction des valeurs
  arma::vec predictions = X_std * coefficients;

  // Retourne les résultats sous forme de liste
  return List::create(Named("coefficients") = coefficients,
                      Named("predictions") = predictions);
}