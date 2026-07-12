/* global L, bootstrap */

document.addEventListener('DOMContentLoaded', () => {
  const config = window.APP_CONFIG || {};
  const map = L.map('map', { zoomControl: true }).setView([config.initialLatitude, config.initialLongitude], 11);

  const tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  });
  tileLayer.addTo(map);

  const state = {
    propertyMarker: null,
    layers: {
      schools: L.layerGroup().addTo(map),
      parks: L.layerGroup().addTo(map),
      hospitals: L.layerGroup().addTo(map),
      subway: L.layerGroup().addTo(map),
      libraries: L.layerGroup().addTo(map),
      banks: L.layerGroup().addTo(map),
      pharmacies: L.layerGroup().addTo(map),
      groceries: L.layerGroup().addTo(map),
      restaurants: L.layerGroup().addTo(map),
    },
  };

  const elements = {
    form: document.getElementById('predictionForm'),
    bedrooms: document.getElementById('bedrooms'),
    bathrooms: document.getElementById('bathrooms'),
    estimatedAreaSqft: document.getElementById('estimated_area_sqft'),
    homeType: document.getElementById('home_type'),
    latitude: document.getElementById('latitude'),
    longitude: document.getElementById('longitude'),
    priceDisplay: document.getElementById('priceDisplay'),
    neighbourhoodDisplay: document.getElementById('neighbourhoodDisplay'),
    pricePerSqft: document.getElementById('pricePerSqft'),
    predictionAccuracy: document.getElementById('predictionAccuracy'),
    homeTypeDisplay: document.getElementById('homeTypeDisplay'),
    timestamp: document.getElementById('predictionTimestamp'),
    errorAlert: document.getElementById('errorAlert'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    resetBtn: document.getElementById('resetBtn'),
    themeToggle: document.getElementById('themeToggle'),
    btnSpinner: document.querySelector('.btn-spinner'),
    btnIcon: document.querySelector('.btn-icon'),
    btnText: document.querySelector('.btn-text'),
    resultCard: document.getElementById('resultCard'),
    distanceSchool: document.getElementById('distance_school'),
    distanceHospital: document.getElementById('distance_hospital'),
    distancePark: document.getElementById('distance_park'),
    distanceLibrary: document.getElementById('distance_library'),
    distanceBank: document.getElementById('distance_bank'),
    distancePharmacy: document.getElementById('distance_pharmacy'),
    distanceGrocery: document.getElementById('distance_grocery'),
    distanceSubway: document.getElementById('distance_subway'),
    restaurants: document.getElementById('restaurants'),
    parks: document.getElementById('parks'),
    schools: document.getElementById('schools'),
  };
  const defaultPredictionAccuracy = (elements.predictionAccuracy?.textContent || '').trim() || '0.00%';

  const currencyFormatter = new Intl.NumberFormat('en-CA', {
    style: 'currency',
    currency: 'CAD',
    maximumFractionDigits: 0,
  });

  function getThemePreference() {
    const stored = localStorage.getItem('toronto-price-ai-theme');
    if (stored) {
      return stored;
    }
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }

  function applyTheme(theme) {
    document.body.classList.toggle('light-mode', theme === 'light');
    localStorage.setItem('toronto-price-ai-theme', theme);
    const icon = elements.themeToggle.querySelector('i');
    const label = elements.themeToggle.querySelector('.toggle-label');
    if (theme === 'light') {
      icon.className = 'bi bi-sun me-1';
      label.textContent = 'Light mode';
    } else {
      icon.className = 'bi bi-moon-stars me-1';
      label.textContent = 'Dark mode';
    }
  }

  function setLoading(isLoading) {
    elements.loadingOverlay.classList.toggle('d-none', !isLoading);
    elements.btnSpinner.classList.toggle('d-none', !isLoading);
    elements.btnText.textContent = isLoading ? 'Predicting...' : 'Predict Price';
    elements.btnIcon.classList.toggle('d-none', isLoading);
    elements.form.querySelectorAll('input, button').forEach((control) => {
      if (control !== elements.resetBtn && control !== elements.themeToggle) {
        control.disabled = isLoading;
      }
    });
  }

  function clearError() {
    elements.errorAlert.classList.add('d-none');
    elements.errorAlert.textContent = '';
  }

  function showError(message) {
    elements.errorAlert.textContent = message;
    elements.errorAlert.classList.remove('d-none');
  }

  function createMarkerIcon(kind, iconClass) {
    return L.divIcon({
      className: '',
      html: `<div class="marker-icon ${kind}"><i class="${iconClass}"></i></div>`,
      iconSize: [42, 42],
      iconAnchor: [21, 42],
      popupAnchor: [0, -38],
    });
  }

  function markerToneForKind(kind) {
    if (kind === 'schools') return 'school';
    if (kind === 'parks') return 'park';
    if (kind === 'hospitals') return 'hospital';
    if (kind === 'subway') return 'subway';
    return 'generic';
  }

  function addPropertyMarker(latitude, longitude) {
    if (state.propertyMarker) {
      state.propertyMarker.remove();
    }

    state.propertyMarker = L.marker([latitude, longitude], {
      icon: createMarkerIcon('property', 'bi-house-door-fill'),
    }).addTo(map);

    state.propertyMarker.bindPopup('Selected property location').openPopup();
  }

  function addAmenityMarkers(kind, items, iconClass) {
    state.layers[kind].clearLayers();
    items.forEach((item) => {
      if (typeof item.latitude !== 'number' || typeof item.longitude !== 'number') {
        return;
      }

      const marker = L.marker([item.latitude, item.longitude], {
        icon: createMarkerIcon(markerToneForKind(kind), iconClass),
      });

      marker.bindPopup(`
        <div class="fw-semibold">${item.name || 'Amenity'}</div>
        <div>${item.distance_km.toFixed(2)} km away</div>
      `);
      marker.addTo(state.layers[kind]);
    });
  }

  function clearAmenityLayers() {
    Object.values(state.layers).forEach((layer) => layer.clearLayers());
  }

  function updateResultCard(data) {
    elements.priceDisplay.textContent = data.predicted_price_formatted || currencyFormatter.format(data.predicted_price || 0);
    elements.neighbourhoodDisplay.textContent = data.neighbourhood || 'Unknown neighbourhood';
    elements.pricePerSqft.textContent = currencyFormatter.format(data.price_per_sqft || 0);
     elements.predictionAccuracy.textContent = data.prediction_accuracy || defaultPredictionAccuracy;
    elements.homeTypeDisplay.textContent = data.home_type || '-';
    elements.timestamp.textContent = data.prediction_timestamp || new Date().toLocaleString();

    elements.distanceSchool.textContent = `${Number(data.distance_school).toFixed(2)} km`;
    elements.distanceHospital.textContent = `${Number(data.distance_hospital).toFixed(2)} km`;
    elements.distancePark.textContent = `${Number(data.distance_park).toFixed(2)} km`;
    elements.distanceLibrary.textContent = `${Number(data.distance_library).toFixed(2)} km`;
    elements.distanceBank.textContent = `${Number(data.distance_bank).toFixed(2)} km`;
    elements.distancePharmacy.textContent = `${Number(data.distance_pharmacy).toFixed(2)} km`;
    elements.distanceGrocery.textContent = `${Number(data.distance_grocery).toFixed(2)} km`;
    elements.distanceSubway.textContent = `${Number(data.distance_subway).toFixed(2)} km`;
    elements.restaurants.textContent = `${Number(data.restaurants)}`;
    elements.parks.textContent = `${Number(data.parks)}`;
    elements.schools.textContent = `${Number(data.schools)}`;

    elements.resultCard.classList.remove('prediction-pulse');
    void elements.resultCard.offsetWidth;
    elements.resultCard.classList.add('prediction-pulse');
  }

  function renderAmenityLayers(data) {
    clearAmenityLayers();

    const nearbyPoints = data.nearby_points || {};
    addAmenityMarkers('schools', nearbyPoints.schools || [], 'bi-mortarboard-fill');
    addAmenityMarkers('parks', nearbyPoints.parks || [], 'bi-tree-fill');
    addAmenityMarkers('hospitals', nearbyPoints.hospitals || [], 'bi-hospital-fill');
    addAmenityMarkers('subway', nearbyPoints.subway || [], 'bi-train-front-fill');
    addAmenityMarkers('libraries', nearbyPoints.libraries || [], 'bi-book-half');
    addAmenityMarkers('banks', nearbyPoints.banks || [], 'bi-cash-coin');
    addAmenityMarkers('pharmacies', nearbyPoints.pharmacies || [], 'bi-capsule-pill');
    addAmenityMarkers('groceries', nearbyPoints.groceries || [], 'bi-basket-fill');
    addAmenityMarkers('restaurants', nearbyPoints.restaurants || [], 'bi-cup-hot-fill');
  }

  function syncInputs(latitude, longitude) {
    elements.latitude.value = Number(latitude).toFixed(6);
    elements.longitude.value = Number(longitude).toFixed(6);
  }

  map.on('click', (event) => {
    syncInputs(event.latlng.lat, event.latlng.lng);
    addPropertyMarker(event.latlng.lat, event.latlng.lng);
  });

  elements.form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();
    setLoading(true);

    try {
      const payload = {
        Bedrooms: Number(elements.bedrooms.value),
        Bathrooms: Number(elements.bathrooms.value),
        EstimatedArea: Number(elements.estimatedAreaSqft.value),
        HomeType: elements.homeType.value,
        Latitude: Number(elements.latitude.value),
        Longitude: Number(elements.longitude.value),
      };

      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed.');
      }

      updateResultCard(data);
      renderAmenityLayers(data);
      addPropertyMarker(payload.Latitude, payload.Longitude);
      map.setView([payload.Latitude, payload.Longitude], 14, { animate: true });
    } catch (error) {
      showError(error.message || 'Unable to generate prediction.');
    } finally {
      setLoading(false);
    }
  });

  elements.resetBtn.addEventListener('click', () => {
    elements.form.reset();
    elements.bedrooms.value = 3;
    elements.bathrooms.value = 2;
    elements.estimatedAreaSqft.value = 1500;
    elements.homeType.value = 'Detached';
    syncInputs(config.initialLatitude, config.initialLongitude);
    elements.priceDisplay.textContent = 'CAD $0';
    elements.neighbourhoodDisplay.textContent = 'Neighbourhood will appear here';
    elements.pricePerSqft.textContent = 'CAD $0';
    elements.predictionAccuracy.textContent = defaultPredictionAccuracy;
    elements.homeTypeDisplay.textContent = '-';
    elements.timestamp.textContent = 'Not generated yet';
    ['distance_school', 'distance_hospital', 'distance_park', 'distance_library', 'distance_bank', 'distance_pharmacy', 'distance_grocery', 'distance_subway'].forEach((id) => {
      document.getElementById(id).textContent = '-';
    });
    ['restaurants', 'parks', 'schools'].forEach((id) => {
      document.getElementById(id).textContent = '-';
    });
    clearError();
    clearAmenityLayers();
    if (state.propertyMarker) {
      state.propertyMarker.remove();
      state.propertyMarker = null;
    }
    map.setView([config.initialLatitude, config.initialLongitude], 11, { animate: true });
  });

  elements.themeToggle.addEventListener('click', () => {
    const nextTheme = document.body.classList.contains('light-mode') ? 'dark' : 'light';
    applyTheme(nextTheme);
  });

  applyTheme(getThemePreference());
  syncInputs(config.initialLatitude, config.initialLongitude);
  addPropertyMarker(config.initialLatitude, config.initialLongitude);
});
