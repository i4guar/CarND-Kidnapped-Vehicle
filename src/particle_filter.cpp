/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>


#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  std::default_random_engine gen;

  num_particles = 1000;  // Set the number of particles
  
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  for(int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    
    particles.push_back(p);
    weights.push_back(p.weight);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  
  for(int i = 0; i < particles.size(); i++) {
    Particle p = particles[i];
    double new_x = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
    double new_y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
    double new_theta = p.theta + yaw_rate * delta_t;
    
    std::normal_distribution<double> dist_x(new_x, std_pos[0]);
    std::normal_distribution<double> dist_y(new_y, std_pos[1]);
    std::normal_distribution<double> dist_theta(new_theta, std_pos[2]);
    
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs> observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  
  for(int i = 0; i < particles.size(); i++) {
    Particle p = particles[i];
    vector<LandmarkObs> global_observations; // in world coordinates
    
    // transform to global observations
    for(int j = 0; j < observations.size(); j++) {
      LandmarkObs obs_local = observations[j];
      LandmarkObs obs_global = LandmarkObs();
      obs_global.x = p.x + (cos(p.theta) * obs_local.x) - (sin(p.theta) * obs_local.y);
      obs_global.y = p.y + (sin(p.theta) * obs_local.x) + (cos(p.theta) * obs_local.y);
      obs_global.id = obs_local.id;
      global_observations.push_back(obs_global);
    }
    
    vector<int> landmark_ids;
    vector<double> sense_x;
    vector<double> sense_y;
    for(int i = 0; predicted.size(); i++) {
      double min_dist = 10000000000;
      LandmarkObs min_landmark;
      LandmarkObs pred = predicted[i];
      for(int j = 0; global_observations.size(); j++) {
        LandmarkObs obs = global_observations[j];
        double landmark_dist = dist(pred.x, pred.y, obs.x, obs.y);
        if(landmark_dist < min_dist) {
          min_dist = landmark_dist;
          min_landmark = obs;
        }
      }
      
      landmark_ids.push_back(min_landmark.id);
      sense_x.push_back(min_landmark.x); // pred.x?
      sense_y.push_back(min_landmark.y); // pred.y?
    }
    
    SetAssociations(p, landmark_ids, sense_x, sense_y);
  
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  vector<LandmarkObs> map_landmark_obs;
  
  for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
    LandmarkObs l = LandmarkObs();
    l.id = map_landmarks.landmark_list[i].id_i;
    l.x = map_landmarks.landmark_list[i].x_f;
    l.y = map_landmarks.landmark_list[i].y_f;
    map_landmark_obs.push_back(l);
  }
  
  dataAssociation(map_landmark_obs, observations);
  
  for(int i = 0; i < particles.size(); i++) {
    Particle p = particles[i];
    
    for (int j = 0; j < p.associations.size(); j++) {
      int landmark_id = p.associations[j];
      LandmarkObs landmark;
      // get landmark with id equals landmark_id
      for (int k = 0; k < map_landmark_obs.size(); k++) {
        if(map_landmark_obs[k].id == landmark_id) {
          landmark = map_landmark_obs[k];
        }
      }
      
      // mulit var
      p.weight *= multiv_prob(std_landmark[0], std_landmark[1], p.sense_x[j], p.sense_y[j], landmark.x, landmark.y);
      weights[i] = p.weight;
    }
  }
  // normalize weights
  double sum_of_weights = std::accumulate(weights.begin(), weights.end(), 0);
  
  for (int i = 0; i < weights.size(); i++) {
    weights[i] /= sum_of_weights;
    particles[i].weight = weights[i];
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

// from udacity classroom
double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}