/* db_js.js - DataBase JavaScript: Helper functions to drive functionality of the DB user interfaces

Functions in this file are relevant for the user interfaces for the database exposed as websites.

The usage of this file assumes awsServices.js is loaded
*/

// CONSTANTS

// FUNCTIONS

function checkState() {
  return;
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
