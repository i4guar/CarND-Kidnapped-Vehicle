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

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  num_particles = 100;  // Set the number of particles
  
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i = 0; i < num_particles; i++) {
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
  
  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) >= 0.00001) {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x += velocity * cos(particles[i].theta) * delta_t;
      particles[i].y += velocity * sin(particles[i].theta) * delta_t;
    }
    std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
    
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs> &observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
 
    for (unsigned int i = 0; i < observations.size(); i++) {
      double min_dist = std::numeric_limits<double>::max();
      LandmarkObs min_landmark;
      LandmarkObs obs = observations[i];
      for (unsigned int j = 0; j < predicted.size(); j++) {
        LandmarkObs pred = predicted[j];
        double landmark_dist = dist(obs.x, obs.y, pred.x, pred.y);
        if(landmark_dist < min_dist) {
          min_dist = landmark_dist;
          min_landmark = pred;
        }
      }
      observations[i].id = min_landmark.id;
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
  for (int i = 0; i < num_particles; i++) {
    vector<LandmarkObs> map_landmark_obs;

    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      LandmarkObs l = LandmarkObs();
      l.id = map_landmarks.landmark_list[j].id_i;
      l.x = map_landmarks.landmark_list[j].x_f;
      l.y = map_landmarks.landmark_list[j].y_f;
      if(dist(particles[i].x, particles[i].y, l.x, l.y) <= sensor_range) {
        map_landmark_obs.push_back(l); 
      }
    }
  
    vector<LandmarkObs> obs_world;
    // transform observations to world coordinates  
    for (unsigned int j = 0; j < observations.size(); j++) {
      LandmarkObs obs_local = observations[j];
      LandmarkObs obs_global = LandmarkObs();
      obs_global.x = particles[i].x + (cos(particles[i].theta) * obs_local.x) - (sin(particles[i].theta) * obs_local.y);
      obs_global.y = particles[i].y + (sin(particles[i].theta) * obs_local.x) + (cos(particles[i].theta) * obs_local.y);
      obs_world.push_back(obs_global);
    }
    
    dataAssociation(map_landmark_obs, obs_world);
    // reset weight
    particles[i].weight = 1.0;
    
    for (unsigned int j = 0; j < obs_world.size(); j++) {
      int landmark_id = obs_world[j].id;
      LandmarkObs landmark;
      landmark.id = -1;
     
      // get landmark with id equals landmark_id
      for (unsigned int k = 0; k < map_landmark_obs.size(); k++) {
        if(map_landmark_obs[k].id == landmark_id) {
          landmark = map_landmark_obs[k];
          break;
        }
      }
     
      if (landmark.id == -1) {
        continue;
      }
      // mulit var
      double w = multiv_prob(std_landmark[0], std_landmark[1], obs_world[j].x, obs_world[j].y, landmark.x, landmark.y);
      particles[i].weight *= w;
      weights[i] = particles[i].weight;
    }
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::random_device rd;
  std::mt19937 gen_i(rd());
  
  std::discrete_distribution<> d(weights.begin(), weights.end());
  vector<Particle> resampled_particles;
  for (int i = 0; i < num_particles; i++) {
    resampled_particles.push_back(particles[d(gen_i)]);
  }
  particles = resampled_particles;
  
  weights.clear();
  for (unsigned int i = 0; i < particles.size(); i++) {
    weights.push_back(particles[i].weight);
  }
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