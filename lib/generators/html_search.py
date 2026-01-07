"""HTML generator for search bar and functionality."""


def generate_search_bar() -> str:
    """Generate HTML for search bar.

    Returns:
        HTML string for search bar
    """
    html = """
    <div class="search-section">
        <div class="search-container">
            <div class="search-icon">🔍</div>
            <input
                type="text"
                id="participant-search"
                class="search-input"
                placeholder="Search by address (0x...) or token ID (123)"
            />
            <button id="search-button" class="search-button">Search</button>
        </div>
        <div id="search-results" class="search-results hidden"></div>
    </div>
    """

    return html


def generate_search_css() -> str:
    """Generate CSS for search bar.

    Returns:
        CSS string
    """
    css = """
    /* Search Section Styles */
    .search-section {
        margin: 30px 0;
    }

    .search-container {
        display: flex;
        align-items: center;
        gap: 12px;
        background: white;
        border: 2px solid #e1e8ed;
        border-radius: 12px;
        padding: 12px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: border-color 0.2s, box-shadow 0.2s;
        max-width: 800px;
        margin: 0 auto;
    }

    .search-container:focus-within {
        border-color: #FF004D;
        box-shadow: 0 4px 12px rgba(230, 0, 74, 0.15);
    }

    .search-icon {
        font-size: 20px;
        color: #7f8c8d;
    }

    .search-input {
        flex: 1;
        border: none;
        outline: none;
        font-size: 16px;
        padding: 8px;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        color: #2c3e50;
    }

    .search-input::placeholder {
        color: #95a5a6;
    }

    .search-button {
        background: #FF004D;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .search-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(230, 0, 74, 0.3);
    }

    .search-button:active {
        transform: translateY(0);
    }

    .search-results {
        margin-top: 20px;
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .search-results.hidden {
        display: none;
    }

    .search-results.empty {
        text-align: center;
        padding: 40px;
        color: #95a5a6;
    }

    .profile-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 25px;
        padding-bottom: 20px;
        border-bottom: 2px solid #e1e8ed;
    }

    .profile-address {
        font-size: 24px;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        color: #2c3e50;
        font-weight: 600;
    }

    .profile-badges {
        display: flex;
        gap: 10px;
    }

    .profile-badge {
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-locker {
        background: rgba(76, 175, 80, 0.2);
        color: #2e7d32;
    }

    .badge-voter {
        background: rgba(33, 150, 243, 0.2);
        color: #1565c0;
    }

    .profile-stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 25px;
    }

    .profile-stat-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e1e8ed;
    }

    .profile-stat-label {
        font-size: 12px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .profile-stat-value {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }

    .profile-stat-secondary {
        font-size: 13px;
        color: #7f8c8d;
        margin-top: 4px;
    }

    .profile-section {
        margin-top: 25px;
    }

    .profile-section-title {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .profile-list {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
    }

    .profile-list-item {
        padding: 8px;
        border-bottom: 1px solid #e1e8ed;
    }

    .profile-list-item:last-child {
        border-bottom: none;
    }

    .profile-list-empty {
        text-align: center;
        padding: 20px;
        color: #95a5a6;
        font-style: italic;
    }

    .token-id-badge {
        display: inline-block;
        background: #FF004D;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        margin: 4px;
    }

    .pool-address-badge {
        display: inline-block;
        background: white;
        border: 1px solid #e1e8ed;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        margin: 4px;
        color: #2c3e50;
    }

    @media (max-width: 768px) {
        .search-container {
            flex-direction: column;
            align-items: stretch;
        }

        .search-button {
            width: 100%;
        }

        .profile-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 15px;
        }

        .profile-stats-grid {
            grid-template-columns: 1fr;
        }
    }
    """

    return css


def generate_search_js() -> str:
    """Generate JavaScript for search functionality.

    Returns:
        JavaScript string
    """
    js = """
    // Search functionality
    const searchButton = document.getElementById('search-button');
    const searchInput = document.getElementById('participant-search');
    const searchResults = document.getElementById('search-results');

    if (searchButton && searchInput && searchResults) {
        // Load participant data from script tag
        const participantsData = window.PARTICIPANTS_DATA || {};

        function performSearch() {
            const query = searchInput.value.trim();
            if (!query) {
                searchResults.classList.add('hidden');
                return;
            }

            // Search by address (case-insensitive partial match)
            if (query.startsWith('0x')) {
                const queryLower = query.toLowerCase();
                const matches = Object.entries(participantsData).filter(
                    ([addr, profile]) => addr.toLowerCase().includes(queryLower)
                );

                if (matches.length === 0) {
                    showNoResults(query);
                } else if (matches.length === 1) {
                    showProfile(matches[0][1]);
                } else {
                    showMultipleResults(matches);
                }
            }
            // Search by token ID (exact match)
            else if (/^\\d+$/.test(query)) {
                const tokenId = parseInt(query);
                const match = Object.values(participantsData).find(
                    profile => profile.token_ids && profile.token_ids.includes(tokenId)
                );

                if (match) {
                    showProfile(match);
                } else {
                    showNoResults(query);
                }
            }
            // Invalid query format
            else {
                searchResults.innerHTML = `
                    <div class="search-results empty">
                        <p>Please enter a valid address (0x...) or token ID (number)</p>
                    </div>
                `;
                searchResults.classList.remove('hidden');
            }
        }

        function showNoResults(query) {
            searchResults.innerHTML = `
                <div class="search-results empty">
                    <p>No participant found for "${escapeHtml(query)}"</p>
                </div>
            `;
            searchResults.classList.remove('hidden');
        }

        function showMultipleResults(matches) {
            const resultsHtml = matches.map(([addr, profile]) => `
                <div class="profile-list-item" onclick="window.searchInput.value='${addr}'; window.performSearch();" style="cursor: pointer;">
                    <code class="address-code">${addr}</code>
                    <span style="margin-left: 10px; color: #7f8c8d;">
                        ${profile.total_locked > 0 ? profile.total_locked.toFixed(4) + ' BTC' : ''}
                        ${profile.current_voting_power > 0 ? ' • ' + profile.current_voting_power.toFixed(2) + ' veBTC' : ''}
                    </span>
                </div>
            `).join('');

            searchResults.innerHTML = `
                <div class="profile-section-title">Found ${matches.length} matches</div>
                <div class="profile-list">${resultsHtml}</div>
            `;
            searchResults.classList.remove('hidden');
        }

        function showProfile(profile) {
            const badges = [];
            if (profile.num_locks > 0) badges.push('<span class="profile-badge badge-locker">Locker</span>');
            if (profile.total_votes_cast > 0) badges.push('<span class="profile-badge badge-voter">Voter</span>');

            const tokenIdsHtml = profile.token_ids && profile.token_ids.length > 0
                ? profile.token_ids.map(id => `<span class="token-id-badge">#${id}</span>`).join('')
                : '<span class="profile-list-empty">No token IDs</span>';

            const poolsHtml = profile.pools_voted && profile.pools_voted.length > 0
                ? profile.pools_voted.map(pool => `<span class="pool-address-badge">${pool.substring(0, 10)}...</span>`).join('')
                : '<span class="profile-list-empty">No pools voted</span>';

            searchResults.innerHTML = `
                <div class="profile-header">
                    <div class="profile-address">${profile.address.substring(0, 10)}...${profile.address.substring(profile.address.length - 8)}</div>
                    <div class="profile-badges">${badges.join('')}</div>
                </div>

                <div class="profile-stats-grid">
                    <div class="profile-stat-card">
                        <div class="profile-stat-label">Total Locked</div>
                        <div class="profile-stat-value">${profile.total_locked ? profile.total_locked.toFixed(4) : '0.0000'}</div>
                        <div class="profile-stat-secondary">${profile.num_locks || 0} transactions</div>
                    </div>

                    <div class="profile-stat-card">
                        <div class="profile-stat-label">Voting Power</div>
                        <div class="profile-stat-value">${profile.current_voting_power ? profile.current_voting_power.toFixed(2) : '0.00'}</div>
                        <div class="profile-stat-secondary">${profile.total_votes_cast || 0} votes cast</div>
                    </div>

                    <div class="profile-stat-card">
                        <div class="profile-stat-label">First Seen</div>
                        <div class="profile-stat-value" style="font-size: 16px;">${profile.first_lock_date || profile.first_vote_date || 'N/A'}</div>
                        <div class="profile-stat-secondary">Last: ${profile.last_lock_date || profile.last_vote_date || 'N/A'}</div>
                    </div>

                    <div class="profile-stat-card">
                        <div class="profile-stat-label">Pools Voted</div>
                        <div class="profile-stat-value">${profile.pools_voted ? profile.pools_voted.length : 0}</div>
                        <div class="profile-stat-secondary">${profile.token_ids ? profile.token_ids.length : 0} token IDs</div>
                    </div>
                </div>

                <div class="profile-section">
                    <div class="profile-section-title">🎟️ Token IDs</div>
                    <div class="profile-list">
                        ${tokenIdsHtml}
                    </div>
                </div>

                <div class="profile-section">
                    <div class="profile-section-title">🏊 Pools Voted</div>
                    <div class="profile-list">
                        ${poolsHtml}
                    </div>
                </div>
            `;
            searchResults.classList.remove('hidden');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Event listeners
        searchButton.addEventListener('click', performSearch);
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                performSearch();
            }
        });

        // Make functions available globally for multiple results onclick
        window.searchInput = searchInput;
        window.performSearch = performSearch;
    }
    """

    return js
