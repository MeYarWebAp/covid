import streamlit as st
lc=['Netherlands','Czechia','Lithuania','Austria','Poland','Slovenia','Estonia','Italy','Slovakia','Ireland','Denmark',
 'Iceland','Cyprus','Greece','Belgium','Bulgaria','France','Germany','Latvia','Spain','Norway','Romania','Liechtenstein',
 'Portugal','Luxembourg','Hungary','Malta','Croatia','Finland','Sweden']
lm=['EntertainmentVenuesPartial','RestaurantsCafesPartial','EntertainmentVenues','MassGatherAll','ClosSec','GymsSportsCentresPartial','ClosPrim',
 'NonEssentialShopsPartial','ClosPubAnyPartial','RestaurantsCafes','GymsSportsCentres','MassGather50','PrivateGatheringRestrictions',
 'MassGatherAllPartial',
 'ClosHigh','NonEssentialShops','ClosSecPartial','OutdoorOver500','ClosDaycare','BanOnAllEvents','IndoorOver500','QuarantineForInternationalTravellers',
 'ClosHighPartial','IndoorOver100','Teleworking','ClosPubAny','PlaceOfWorshipPartial','MasksMandatoryClosedSpacesPartial','MassGather50Partial',
 'StayHomeOrderPartial','OutdoorOver100','IndoorOver50','ClosPrimPartial','PrivateGatheringRestrictionsPartial','MasksMandatoryClosedSpaces',
 'OutdoorOver1000','TeleworkingPartial','MasksMandatoryAllSpaces','OutdoorOver50','StayHomeOrder','QuarantineForInternationalTravellersPartial',
 'MasksMandatoryAllSpacesPartial','StayHomeGen','PlaceOfWorship','ClosDaycarePartial','IndoorOver1000','BanOnAllEventsPartial',
 'HotelsOtherAccommodationPartial',
 'StayHomeRiskG','ClosureOfPublicTransportPartial','AdaptationOfWorkplace','HotelsOtherAccommodation','MasksVoluntaryClosedSpacesPartial',
 'RegionalStayHomeOrderPartial','AdaptationOfWorkplacePartial','MasksVoluntaryAllSpaces','MasksVoluntaryAllSpacesPartial','MasksVoluntaryClosedSpaces',
 'SocialCircle','WorkplaceClosures','RegionalStayHomeOrder','ClosureOfPublicTransport','StayHomeGenPartial','WorkplaceClosuresPartial',
 'StayHomeRiskGPartial','SocialCirclePartial']
st.multiselect('select a country or a set of countries',lc)
st.multiselect('select a government response or a set of government responses',lm)
st.number_input('select number of days per incidence')
st.selectbox('select a method',['hierarchical method'])
st.button('posterior predictive analysis')
