$(document).ready(function() {
    // Initialize star ratings
    initializeStarRatings();

    // Function to initialize star rating interactions
    function initializeStarRatings() {
        // Hover effect
        $('.star-rating .star').on('mouseenter', function() {
            const rating = $(this).data('rating');
            const stars = $(this).parent().find('.star');
            
            // Add hover class to current star and all stars before it
            stars.each(function(index) {
                if (index < rating) {
                    $(this).addClass('hover');
                    $(this).find('i').removeClass('far').addClass('fas');
                } else {
                    $(this).removeClass('hover');
                    if (!$(this).hasClass('active')) {
                        $(this).find('i').removeClass('fas').addClass('far');
                    }
                }
            });
        });

        // Remove hover effect when mouse leaves
        $('.star-rating').on('mouseleave', function() {
            const stars = $(this).find('.star');
            
            stars.removeClass('hover');
            stars.each(function() {
                if (!$(this).hasClass('active')) {
                    $(this).find('i').removeClass('fas').addClass('far');
                }
            });
        });

        // Handle click for rating
        $('.star-rating .star').on('click', function() {
            const rating = $(this).data('rating');
            const starRating = $(this).parent();
            const movieId = starRating.data('movie-id');
            const movieTitle = starRating.data('movie-title');
            const movieYear = starRating.data('movie-year');
            
            // Set active class on clicked star and all stars before it
            const stars = starRating.find('.star');
            stars.removeClass('active');
            stars.find('i').removeClass('fas').addClass('far');
            
            stars.each(function(index) {
                if (index < rating) {
                    $(this).addClass('active');
                    $(this).find('i').removeClass('far').addClass('fas');
                }
            });
            
            // Submit rating to the server
            submitRating(movieId, movieTitle, movieYear, rating);
        });
    }

    // Function to submit rating to the server
    function submitRating(movieId, movieTitle, movieYear, rating) {
        // Show loading indicator
        showLoading();
        
        // Send rating to server
        $.ajax({
            url: '/rate',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                imdb_id: movieId,
                title: movieTitle,
                year: movieYear,
                rating: rating
            }),
            success: function(response) {
                // Hide loading
                hideLoading();
                
                // If successful, update recommendations
                if (response.success) {
                    updateMovies(response.movies);
                }
            },
            error: function(error) {
                // Hide loading
                hideLoading();
                console.error('Error submitting rating:', error);
            }
        });
    }

    // Function to update movies in the UI
    function updateMovies(movies) {
        const container = $('#movies-container');
        container.empty();
        
        // Add new movies
        movies.forEach(function(movie) {
            const movieHtml = createMovieCard(movie);
            container.append(movieHtml);
        });
        
        // Re-initialize star ratings for new movie cards
        initializeStarRatings();
    }

    // Function to create a movie card
    function createMovieCard(movie) {
        // Extract year from date
        let year = '';
        if (movie.DatePublished) {
            year = movie.DatePublished.split('-')[0];
        }
        
        // Process genres
        let genresHtml = '';
        if (movie.Genres) {
            const genres = movie.Genres.split(',');
            genres.forEach(function(genre) {
                genresHtml += `<span class="badge bg-primary">${genre.trim()}</span> `;
            });
        }
        
        // Truncate description
        let description = movie.Description || '';
        if (description.length > 200) {
            description = description.substring(0, 200) + '...';
        }
        
        // Build HTML for the movie card
        return `
            <div class="col-md-6 mb-4 movie-card">
                <div class="card h-100">
                    <div class="row g-0">
                        <div class="col-md-4">
                            <img src="${movie.PosterLink}" class="img-fluid rounded-start movie-poster" alt="${movie.Name}">
                        </div>
                        <div class="col-md-8">
                            <div class="card-body">
                                <h5 class="card-title">${movie.Name}</h5>
                                <p class="card-text">
                                    <small class="text-muted">
                                        ${year}
                                        ${movie.Director ? `&bull; ${movie.Director}` : ''}
                                    </small>
                                </p>
                                <p class="card-text genres">
                                    ${genresHtml}
                                </p>
                                <p class="card-text cast">
                                    ${movie.Actors ? `<strong>Cast:</strong> ${movie.Actors}` : ''}
                                </p>
                                <p class="card-text description">${description}</p>
                                <div class="rating-container">
                                    <p><strong>Rate this movie:</strong></p>
                                    <div class="star-rating" data-movie-id="${movie.id}" data-movie-title="${movie.Name}" data-movie-year="${year}">
                                        <span class="star" data-rating="1"><i class="far fa-star"></i></span>
                                        <span class="star" data-rating="2"><i class="far fa-star"></i></span>
                                        <span class="star" data-rating="3"><i class="far fa-star"></i></span>
                                        <span class="star" data-rating="4"><i class="far fa-star"></i></span>
                                        <span class="star" data-rating="5"><i class="far fa-star"></i></span>
                                        <span class="star" data-rating="6"><i class="far fa-star"></i></span>
                                        <span class="star" data-rating="7"><i class="far fa-star"></i></span>
                                        <span class="star" data-rating="8"><i class="far fa-star"></i></span>
                                        <span class="star" data-rating="9"><i class="far fa-star"></i></span>
                                        <span class="star" data-rating="10"><i class="far fa-star"></i></span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Loading indicator functions
    function showLoading() {
        // If loading indicator doesn't exist, create it
        if ($('.loading').length === 0) {
            const loadingHtml = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                </div>
            `;
            $('body').append(loadingHtml);
        }
        $('.loading').show();
    }

    function hideLoading() {
        $('.loading').hide();
    }
}); 