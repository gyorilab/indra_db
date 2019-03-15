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

function _toCookieString(queryDict) {
  console.log('function _toCookieString(queryDict)')
  var cookieString = ''
  for (key in queryDict) {
    if (cookieString.length == 0) cookieString = key + '_eq_' + queryDict[key];
  else cookieString = cookieString + '_and_' + key + '_eq_' + queryDict[key];
  }
  return cookieString;
}

function _fromCookieStringToQuery(cookieString) {
  return cookieString.replace(/_eq_/g, '=').replace(/_and_/g, '&');
}
