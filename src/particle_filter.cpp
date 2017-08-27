/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

// number of particles
#define NUM_PARTICLES   500;

// Verbose level (0 = none, 1 = basic debug info, 2 = details debug info)
#define VERBOSE_LEVEL   0

// if defined, show associated observations in simulator
#undef SHOW_ASSOCIATED_OBS_IN_SIMULATOR


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  // create normal (Gaussian) distribution for x, y and theta
  random_device rd;
  mt19937 gen(rd());
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // initialize particles
  num_particles_ = NUM_PARTICLES;
  
  for (int i=0; i < num_particles_; ++i) {
    Particle particle;
    
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0f;

    particles_.push_back(particle);
    weights_.push_back(1.0);
  }
  
  is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


  // create normal (Gaussian) distribution for x, y and theta
  random_device rd;
  mt19937 gen(rd());

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // predict all particles for the next time step
  for (int i=0; i < num_particles_; ++i) {
  
    double theta = particles_[i].theta;
    
    // check for yaw rate equals zero
    if (fabs(yaw_rate) >= 1e-6) {
      particles_[i].x += velocity / yaw_rate * ( sin(theta + yaw_rate * delta_t) - sin(theta) );
      particles_[i].y += velocity / yaw_rate * ( cos(theta) - cos(theta + yaw_rate * delta_t) );
      particles_[i].theta += yaw_rate * delta_t;
    } else {
      particles_[i].x += velocity * delta_t * cos(theta);
      particles_[i].y += velocity * delta_t * sin(theta);
    }
    
    particles_[i].x += dist_x(gen);
    particles_[i].y += dist_y(gen);
    particles_[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  // Brute force method
  // Complexity inner for loop: n * O(1) = O(n)
  // Complexits outer for loop: n * O(n) = O(n^2)
  
  for(int i=0; i < observations.size(); ++i) {

    double min_dist = __DBL_MAX__;
    int index_min_dist = -1;
    
    for (int j=0; j < predicted.size(); ++j) {
      
      // calculate euclidean distance between prediction and observation
      double euclidean_dist = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      
      // found closer observation
      if (euclidean_dist < min_dist) {
        min_dist = euclidean_dist;
        index_min_dist = j;
      }
    }
    
    // update matched landmark observation
    observations[i].id = index_min_dist;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  if (VERBOSE_LEVEL >= 1) {
    cout << "### Updated weights: ###" << endl;
  }
  
  for (int i=0; i < num_particles_; ++i) {
    
    const double p_x = particles_[i].x;
    const double p_y = particles_[i].y;
    const double p_theta = particles_[i].theta;
    particles_[i].weight = 1.0;
    weights_[i] = 1.0;

#ifdef SHOW_ASSOCIATED_OBS_IN_SIMULATOR
    // reset debug information of particle
    particles_[i].associations.clear();
    particles_[i].sense_x.clear();
    particles_[i].sense_y.clear();
#endif

    //
    // Step 1: Transform observations into map coordinates
    //
    vector<LandmarkObs> map_observations;
    
    for (int j=0; j < observations.size(); ++j) {
      double map_obs_x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
      double map_obs_y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;
      
      LandmarkObs map_obs = {observations[i].id, map_obs_x, map_obs_y};
      map_observations.push_back(map_obs);
    }
    
    //
    // Step 2: Get all landmarks within the sensor range
    //
    vector<LandmarkObs> landmarks_in_range;
    
    for (int j=0; j < map_landmarks.landmark_list.size(); ++j) {
      
      double euclidean_dist = dist(p_x, p_y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      
      if (euclidean_dist <= sensor_range) {
        LandmarkObs l = {
          map_landmarks.landmark_list[j].id_i,
          map_landmarks.landmark_list[j].x_f,
          map_landmarks.landmark_list[j].y_f
        };
        
        landmarks_in_range.push_back(l);
      }
    }
    
    //
    // Step 3: Associate observations to landmarks within the sensor range
    //
    if (landmarks_in_range.size() == 0) {
      cout << "WARNING: Particle(" << i << "): No landmarks in range of " << sensor_range << " m " << endl;
    }
    
    dataAssociation(landmarks_in_range, map_observations);
    
    //
    // Step 4: Update the particles' weight
    //
    double w = 1.0;
    
    for (int j=0; j < map_observations.size(); ++j) {
      
      int obs_id = map_observations[j].id;
      double obs_x = map_observations[j].x;
      double obs_y = map_observations[j].y;
      
      double pred_x = landmarks_in_range[obs_id].x;
      double pred_y = landmarks_in_range[obs_id].y;
      
      // calculate multivariate gaussian weight
      double a = (2 * M_PI * std_landmark[0] * std_landmark[1]);
      double b = pow(obs_x - pred_x, 2) / (2 * pow(std_landmark[0], 2));
      double c = pow(obs_y - pred_y, 2) / (2 * pow(std_landmark[1], 2));
      double p = exp(-(b + c)) / a;
  
      w *= p;

#ifdef SHOW_ASSOCIATED_OBS_IN_SIMULATOR
      // add debug information to particle
      particles_[i].associations.push_back(landmarks_in_range[obs_id].id);
      particles_[i].sense_x.push_back(obs_x);
      particles_[i].sense_y.push_back(obs_y);
#endif

      if (VERBOSE_LEVEL >= 2) {
        cout << "  Associated: " << obs_id << ": x=" << obs_x << " y=" << obs_y
             << " landmark: x=" << landmarks_in_range[obs_id].x << " y=" << landmarks_in_range[obs_id].y
             << " | p=" << p << " w=" << w << endl;
      }
    }
    
    particles_[i].weight = w;
    weights_[i] = w;
    
    if (VERBOSE_LEVEL >= 2) {
      cout << " " << particles_[i].id << ": x=" << particles_[i].x << " y=" << particles_[i].y << " theta=" << particles_[i].theta * 180.0f / M_PI << " w=" << particles_[i].weight << endl;
    }
  }
  
  // normalize weights
  double sum_w = accumulate(weights_.begin(), weights_.end(), 0.0);
  
  for (int i=0; i < num_particles_; ++i) {
    particles_[i].weight /= sum_w;
    weights_[i] /= sum_w;
    
    if (VERBOSE_LEVEL >= 1) {
      cout << " " << particles_[i].id << ": x=" << particles_[i].x << " y=" << particles_[i].y
           << " theta=" << particles_[i].theta * 180.0f / M_PI << " w_normed=" << particles_[i].weight << endl;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  // apply resampling wheel algorithm
  vector<Particle> resampled_particles;
  
  random_device rd;
  mt19937 gen_index(rd());
  uniform_int_distribution<> dist_index(0, num_particles_ - 1);
  int index = dist_index(gen_index);
  
  double beta = 0.0;
  double w_max = *max_element(weights_.begin(), weights_.end());
  mt19937 gen_beta(rd());
  uniform_real_distribution<> dist_beta(0.0, 2.0 * w_max);
  
  for (int i=0; i < num_particles_; ++i) {
    beta += dist_beta(gen_beta);
    
    while (weights_[index] < beta) {
      beta -= weights_[index];
      index = (index + 1) % num_particles_;
    }
    
    Particle p = {
      i,
      particles_[index].x,
      particles_[index].y,
      particles_[index].theta,
      particles_[index].weight,
      particles_[index].associations,
      particles_[index].sense_x,
      particles_[index].sense_y
    };
    
    resampled_particles.push_back(p);
  }
  
  particles_ = resampled_particles;
  
  if (VERBOSE_LEVEL >= 1) {
    cout << "### Resampled particles: ###" << endl;
    
    for(auto& p : particles_) {
      cout << " " << p.id << ": x=" << p.x << " y=" << p.y << " theta=" << p.theta * 180.0f / M_PI << " w=" << p.weight << endl;
    }
  }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
