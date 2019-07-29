/* db_js.js - DataBase JavaScript: Helper functions to drive functionality of the DB user interfaces

Functions in this file are relevant for the user interfaces for the database exposed as websites.

The usage of this file assumes awsServices.js is loaded
*/

// CONSTANTS
SEARCH_STATE_COOKIE_NAME = 'indraDBsearchQuery'
BUTTON_ID = NOTIFY_TAG_ID;

// FUNCTIONS
function checkState() {
  // Get query dict
  let dict_split = getDictFromUrl(window.location.href)
  if (dict_split) {
    let dict = dict_split[0]

    // Check if query string (?) or fragment (#)
    if (dict_split[1] == '?') {
      // Save query string in cookie
      _writeCookie(SEARCH_STATE_COOKIE_NAME, _dictToCookieString(dict), 1)

      // Set button to login redirect
      let buttonTag = document.getElementById(BUTTON_ID)
      console.log(buttonTag)
      buttonTag.textContent = 'Login to continue search';
      buttonTag.onclick = goToLogin;

    } else if (dict_split[1] == '#') {
      // Fragment handling - assume redirect from cognito

      // Get tokens
      let accTokenString = dict['access_token']
      let idTokenString = dict['id_token']

      // Verifies the user and saves cookies
      verifyUser(accTokenString, idTokenString, false, null, 'Continue search')

      // Set button to continue search
      let buttonTag = document.getElementById(BUTTON_ID)
      buttonTag.textContent = 'Continue search';
      buttonTag.onclick = continueSearch;
    }
  }
  return;
}

function goToLogin() {
  // Split off the query string
  getTokenFromAuthEndpoint(window.location.href.split('?')[0])
}

function continueSearch() {
  // Get query cookie
  let cookieQueryString = _readCookie(SEARCH_STATE_COOKIE_NAME)
  let queryDict = _cookieStringToDict(cookieQueryString);
  let endpoint = queryDict['endpoint']
  if (delete queryDict['endpoint']) {

    // Create full URL of previous search
    let full_url = endpoint + _cookieStringToQuery(cookieQueryString, true)
    
    // Delete query cookie
    console.log('Deleting search state cookie')
    _deleteCookie(SEARCH_STATE_COOKIE_NAME)

    // redirect to endpoint
    console.log('redirecting to endpoint')
    console.log(full_url)
     window.location.replace(full_url) // Redirect
  }
}

function _urlToCookieUrl(endpoint) {
  // See: https://stackoverflow.com/questions/1969232/allowed-characters-in-cookies
  // Allowed chars: [a-zA-Z0-9] and !#$%&'*+-.^_`|~
  // Need to replace ':' and '/'
  return endpoint.replace(/:/g, '!').replace(/\//g, '|');
}

function _cookieUrlToUrl(cookieUrl) {
  return cookieUrl.replace(/!/g, ':').replace(/\|/g, '/')
}

function _dictToCookieString(queryDict) {
  console.log('function _dictToCookieString(queryDict)')
  var cookieString = ''
  for (key in queryDict) {
    if (cookieString.length == 0) cookieString = key + '_eq_' + _urlToCookieUrl(queryDict[key]);
  else cookieString = cookieString + '_and_' + key + '_eq_' + _urlToCookieUrl(queryDict[key]);
  }
  return cookieString;
}

function _cookieStringToDict(cookieString) {
  let dict = {}
  for (par of _cookieUrlToUrl(cookieString).split('_and_')) {
    k_v = par.split('_eq_')
    dict[k_v[0]] = k_v[1]
  }
  return dict
}

function _cookieStringToQuery(cookieString, stripEndpoint) {
  let cookieDict = _cookieStringToDict(cookieString)
  let query = ''
  for (key in cookieDict) {
    if (stripEndpoint && key == 'endpoint') {
      continue;
    }
    if (query.length == 0) query = '?' + key + '=' + cookieDict[key];
    else query = query + '&' + key + '=' + cookieDict[key]; 
  }
  return query;
}
